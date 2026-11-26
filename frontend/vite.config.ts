import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In development the React app runs on :5173 and forwards API + image requests to the Node backend.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:4000",
      "/media": "http://localhost:4000",
    },
  },
});
