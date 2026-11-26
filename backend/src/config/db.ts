import pg from "pg";
import { env } from "./env.js";

// NUMERIC -> JS number (prices are small, safe as float)
pg.types.setTypeParser(1700, (v) => (v === null ? null : Number(v)));

export const pool = new pg.Pool({
  host: env.POSTGRES_HOST,
  port: env.POSTGRES_PORT,
  database: env.POSTGRES_DB,
  user: env.POSTGRES_USER,
  password: env.POSTGRES_PASSWORD,
  max: 10,
  connectionTimeoutMillis: 5000,
});

/** Always use $1, $2 placeholders - never string-concatenate user input into SQL. */
export async function query<T extends pg.QueryResultRow = pg.QueryResultRow>(text: string, params: unknown[] = []) {
  return pool.query<T>(text, params);
}

export async function withTransaction<T>(fn: (client: pg.PoolClient) => Promise<T>): Promise<T> {
  const client = await pool.connect();
  try {
    await client.query("BEGIN");
    const out = await fn(client);
    await client.query("COMMIT");
    return out;
  } catch (err) {
    await client.query("ROLLBACK");
    throw err;
  } finally {
    client.release();
  }
}
