/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev, /api is proxied to the FastAPI server (uv run weldsight-serve ...).
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": process.env.WELDSIGHT_API ?? "http://127.0.0.1:8000" } },
  test: { environment: "node" },
});
