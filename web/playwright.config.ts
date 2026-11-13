import { existsSync } from "node:fs";
import { defineConfig, devices } from "@playwright/test";

/**
 * End-to-end tests run against the REAL backend and real Metrica data.
 *
 * Both servers are started automatically unless they are already running:
 *   - FastAPI on :8765 (needs FOOTYMIND_DATA_DIR pointing at processed data)
 *   - Vite dev server on :5173 (proxies /api to :8765 via FOOTYMIND_API)
 *
 * In sandboxes with a preinstalled Chromium, set PW_CHROMIUM or rely on the
 * /opt/pw-browsers fallback below. Never run `playwright install` here.
 */
const preinstalled = process.env.PW_CHROMIUM ?? "/opt/pw-browsers/chromium";
const executablePath = !process.env.PLAYWRIGHT_BROWSERS_PATH && existsSync(preinstalled) ? preinstalled : undefined;

export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [["list"]],
  outputDir: "test-results",
  use: {
    baseURL: process.env.E2E_BASE_URL ?? "http://127.0.0.1:5173",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    launchOptions: executablePath ? { executablePath } : {},
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"], deviceScaleFactor: 1 } }],
  webServer: process.env.E2E_BASE_URL
    ? undefined
    : [
        {
          command: "uv run uvicorn footymind.api.app:app --port 8765",
          cwd: "..",
          url: "http://127.0.0.1:8765/api/health",
          reuseExistingServer: true,
          timeout: 120_000,
        },
        {
          command: "npm run dev -- --port 5173 --strictPort",
          env: { FOOTYMIND_API: "http://127.0.0.1:8765" },
          url: "http://127.0.0.1:5173",
          reuseExistingServer: true,
          timeout: 60_000,
        },
      ],
});
