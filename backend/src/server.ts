import { createApp } from "./app.js";
import { pool } from "./config/db.js";
import { env } from "./config/env.js";

const app = createApp();
const server = app.listen(env.BACKEND_PORT, () => {
  console.log(`StyleFind API listening on http://localhost:${env.BACKEND_PORT}  (AI service: ${env.AI_SERVICE_URL})`);
});
server.requestTimeout = env.AI_TIMEOUT_MS + 10_000;

function shutdown() {
  server.close(() => pool.end().then(() => process.exit(0)));
}
process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
