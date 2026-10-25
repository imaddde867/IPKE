// Optional browser check: node demo/tests/browser.cjs (requires Playwright)
// Start: python -m http.server 8765 --directory demo
const assert = require("node:assert/strict");
const { chromium } = require("playwright");
(async () => {
  const browser = await chromium.launch({
    headless: true,
    executablePath: process.env.CHROMIUM_PATH,
    args: ["--no-sandbox", "--disable-dev-shm-usage"],
  });
  try {
    const page = await browser.newPage({
      viewport: { width: 1440, height: 1000 },
    });
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto(process.env.DEMO_URL || "http://127.0.0.1:8765");
    await page.locator("#console").waitFor({ state: "visible" });
    assert.equal(await page.locator("#graph .step").count(), 14);
    assert.equal(await page.locator("#graph .constraint").count(), 15);
    for (const id of Array.from({ length: 15 }, (_, i) => `C${i + 1}`)) {
      const button = page.locator(`#graph [data-item="${id}"]`);
      await button.click();
      const quote = (await button.textContent())
        .split(" · ")
        .slice(1)
        .join(" · ");
      assert.equal(await page.locator("#source mark").textContent(), quote);
      assert.equal(await button.getAttribute("aria-pressed"), "true");
    }
    await page.locator("#summary-toggle").focus();
    await page.keyboard.press("Enter");
    assert.equal(await page.locator("#graph").isVisible(), false);
    assert.equal(
      await page.locator("#summary details").evaluate((node) => node.open),
      false,
    );
    assert.equal(
      await page.locator("#summary .constraint").first().isVisible(),
      false,
    );
    await page.locator("#summary details summary").click();
    assert.equal(
      await page.locator("#summary .constraint").first().isVisible(),
      true,
    );
    await page.locator("#summary-toggle").click();
    assert.equal(await page.locator("#graph").isVisible(), true);
    const s12row = page
      .locator(".graph-row")
      .filter({ has: page.locator('[data-item="S12"]') });
    assert.match(await s12row.textContent(), /procedure-level C15/);
    await page.locator('#graph [data-item="S12"]').click();
    assert.match(
      await page.locator("#inspector").textContent(),
      /Procedure-level constraint: C15/,
    );
    assert.match(
      await page.locator("#inspector").textContent(),
      /After completing the calibration/,
    );
    await page.getByRole("button", { name: "02 Annotation audit" }).click();
    assert.equal(await page.locator("#human-decisions .decision").count(), 8);
    assert.equal(await page.locator("#decisions .decision").count(), 13);
    await page
      .getByRole("button", { name: "Inspect review:C2 source", exact: true })
      .click();
    assert.equal(await page.locator("#graph").isVisible(), true);
    assert.match(
      await page.locator("#source mark").textContent(),
      /without the charger/,
    );
    await page.getByRole("button", { name: "02 Annotation audit" }).click();
    await page
      .getByRole("button", { name: "Inspect legacy:S12 source", exact: true })
      .click();
    assert.equal(await page.locator("#graph").isVisible(), true);
    assert.match(
      await page.locator("#inspector").textContent(),
      /section-level provenance only/,
    );
    assert.equal(await page.locator("#source mark").count(), 0);
    await page.getByRole("button", { name: "02 Annotation audit" }).click();
    await page
      .getByRole("button", { name: "Inspect legacy:C2 source", exact: true })
      .click();
    assert.match(
      await page.locator("#inspector").textContent(),
      /Legacy candidate · unverified/,
    );
    assert.equal(
      await page.locator('#graph [aria-pressed="true"]').count(),
      0,
    );
    await page.getByRole("button", { name: "03 Corpus status" }).click();
    assert.equal(await page.locator("#corpus .decision").count(), 8);
    assert.match(
      await page.locator("#corpus-status").textContent(),
      /^0 of 8 corpus documents are human verified\.$/,
    );
    const downloadPromise = page.waitForEvent("download");
    await page.getByRole("link", { name: "Export snapshot" }).click();
    assert.equal(
      (await downloadPromise).suggestedFilename(),
      "ipke-epa-snapshot.json",
    );
    await page.getByRole("button", { name: "01 Graph & source" }).click();
    await page.screenshot({ path: "/tmp/ipke-desktop.png", fullPage: true });
    await page.setViewportSize({ width: 390, height: 844 });
    assert.equal(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
      true,
    );
    await page.locator('#graph [data-item="C15"]').click();
    assert.match(
      await page.locator("#inspector").textContent(),
      /S12, S13, S14/,
    );
    await page.screenshot({ path: "/tmp/ipke-mobile.png", fullPage: true });
    assert.deepEqual(errors, []);
    console.log(
      "Desktop/mobile graph, 15 source spans, keyboard toggle, audit, corpus and export passed.",
    );
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
