/**
 * Create or reset an admin account.
 *   npm run create-admin -- --email admin@stylefind.local --password "a long password"
 */
import bcrypt from "bcryptjs";
import { pool, query } from "../config/db.js";

function arg(name: string): string | undefined {
  const i = process.argv.indexOf(`--${name}`);
  return i > -1 ? process.argv[i + 1] : undefined;
}

const email = arg("email")?.trim().toLowerCase();
const password = arg("password");
if (!email || !password || password.length < 10) {
  console.error('Usage: npm run create-admin -- --email you@example.com --password "at least 10 characters"');
  process.exit(1);
}
const hash = await bcrypt.hash(password, 12);
await query(
  `INSERT INTO admin_users (email, password_hash) VALUES ($1, $2)
   ON CONFLICT (email) DO UPDATE SET password_hash = EXCLUDED.password_hash`,
  [email, hash],
);
console.log(`Admin account ready: ${email}`);
await pool.end();
