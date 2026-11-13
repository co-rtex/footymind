/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The API is always addressed at the relative path /api. In development Vite
// proxies it to the FastAPI backend. In production FastAPI serves web/dist
// itself, so the same relative path works with no configuration.
const API_TARGET = process.env.FOOTYMIND_API ?? "http://127.0.0.1:8765";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: false,
    proxy: { "/api": { target: API_TARGET, changeOrigin: true } },
  },
  preview: {
    port: 4173,
    proxy: { "/api": { target: API_TARGET, changeOrigin: true } },
  },
  build: { target: "es2022", sourcemap: true },
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    include: ["src/**/*.test.{ts,tsx}"],
    css: false,
  },
});
