import { defineConfig } from '@playwright/test';

const external = process.env.BLAZE_TEST_URL;
export default defineConfig({
  testDir: './tests', timeout: 90_000, expect: { timeout: 15_000 }, workers: 1,
  forbidOnly: !!process.env.CI, retries: process.env.CI ? 1 : 0,
  outputDir: './test-results', reporter: 'list',
  use: { baseURL: external ?? 'http://127.0.0.1:3211', trace: 'retain-on-failure', screenshot: 'only-on-failure' },
  projects: ['chromium', 'firefox', 'webkit'].map(browserName => ({ name: browserName, use: { browserName: browserName as 'chromium' | 'firefox' | 'webkit', launchOptions: browserName === 'webkit' && process.env.BLAZE_WEBKIT_EXECUTABLE ? { executablePath: process.env.BLAZE_WEBKIT_EXECUTABLE } : undefined } })),
  webServer: external ? undefined : { command: 'node scripts/serve-export.mjs', url: `http://127.0.0.1:3211${process.env.NEXT_BASE_PATH ?? ''}/`, reuseExistingServer: !process.env.CI },
});
