// Headless smoke test: does the app actually boot in a real browser?
//
// Chromium's fake capture device produces a synthetic pattern with no face in
// it, so this cannot check accuracy. What it does check is everything that
// only fails at runtime: module graph resolution, MediaPipe wasm + model
// loading, getUserMedia, the detect loop, and the "no face" UI path.
//
// Run `npm run setup` first so the MediaPipe assets are served locally; the
// test then needs no network at all.
//
// Usage: node test/e2e/smoke.mjs [baseUrl]

import { chromium } from 'playwright';

const BASE = process.argv[2] ?? 'http://localhost:8099';

const failures = [];
const note = (ok, label, detail = '') => {
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}${detail ? ` — ${detail}` : ''}`);
  if (!ok) failures.push(label);
};

const browser = await chromium.launch({
  args: [
    '--use-fake-device-for-media-stream',
    '--use-fake-ui-for-media-stream',
  ],
});

const context = await browser.newContext({
  permissions: ['camera'],
  viewport: { width: 1280, height: 800 },
});

const consoleErrors = [];
const pageErrors = [];
context.on('console', (m) => {
  if (m.type() === 'error') consoleErrors.push(m.text());
});
context.on('weberror', (e) => pageErrors.push(String(e.error())));

const page = await context.newPage();
page.on('pageerror', (e) => pageErrors.push(String(e)));

try {
  await page.goto(BASE, { waitUntil: 'load', timeout: 30000 });
  note(true, 'index.html loads');

  note(
    (await page.title()) === 'Webcam Eye Tracker',
    'page title is set',
    await page.title()
  );

  // The module graph is only exercised once app.js runs, so a bad import shows
  // up as the start button never becoming interactive.
  await page.waitForSelector('#btnStart:not([disabled])', { timeout: 10000 });
  note(true, 'ES module graph resolves and the UI enables');

  const before = await page.textContent('#status');
  await page.click('#btnStart');

  // MediaPipe pulls several MB from a CDN on first run.
  await page.waitForFunction(
    () => document.getElementById('btnStart').textContent.includes('停止'),
    { timeout: 90000 }
  );
  note(true, 'camera starts and the model loads', `status was: ${before?.trim()}`);

  await page.waitForFunction(
    () => {
      const fps = document.getElementById('statFps').textContent;
      return fps !== '-' && Number(fps) > 0;
    },
    { timeout: 30000 }
  );
  const fps = await page.textContent('#statFps');
  note(true, 'the detect loop is running', `${fps} fps on the fake device`);

  // No real face in the synthetic stream, so this is the expected branch.
  const hint = (await page.textContent('#previewHint'))?.trim();
  note(hint === '顔が検出できません', 'the no-face path renders', hint);

  const calibrateDisabled = await page.isDisabled('#btnCalibrate');
  note(!calibrateDisabled, 'calibration becomes available once the camera runs');

  await page.screenshot({ path: 'test/e2e/screenshot-app.png' });

  // The embed example shares the module graph; a broken export breaks it too.
  const embed = await context.newPage();
  const embedErrors = [];
  embed.on('pageerror', (e) => embedErrors.push(String(e)));
  await embed.goto(`${BASE}/examples/embed.html`, { waitUntil: 'load', timeout: 30000 });
  await embed.waitForSelector('#start', { timeout: 10000 });
  note(embedErrors.length === 0, 'examples/embed.html loads clean', embedErrors[0] ?? '');
  await embed.screenshot({ path: 'test/e2e/screenshot-embed.png' });

  // The numeric core is pure ES modules, so it must behave identically in a
  // browser and in Node. Run a miniature calibration in page context.
  const err = await page.evaluate(async () => {
    const [{ buildFeatures, IDX }, { GazeModel }] = await Promise.all([
      import('./js/features.js'),
      import('./js/gaze.js'),
    ]);
    const face = (gx, gy) => {
      const lm = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
      const put = (i, x, y) => (lm[i] = { x, y, z: 0 });
      for (const [eye, cx] of [[IDX.eyeA, 0.43], [IDX.eyeB, 0.57]]) {
        put(eye.outer, cx + (cx < 0.5 ? -0.03 : 0.03), 0.45);
        put(eye.inner, cx + (cx < 0.5 ? 0.03 : -0.03), 0.45);
        put(eye.upper, cx, 0.435);
        put(eye.lower, cx, 0.465);
        put(eye.iris, cx + 0.013 * gx, 0.45 + 0.009 * gy);
      }
      put(IDX.nose, 0.5, 0.55);
      put(IDX.chin, 0.5, 0.75);
      put(IDX.brow, 0.5, 0.28);
      return lm;
    };
    const vec = (x, y) =>
      buildFeatures(face((x / 1000) * 2 - 1, (y / 800) * 2 - 1), 16 / 9).vec;

    const model = new GazeModel();
    let group = 0;
    for (const x of [100, 500, 900]) {
      for (const y of [80, 400, 720]) {
        for (let i = 0; i < 10; i++) model.addSample(vec(x, y), x, y, { group });
        group++;
      }
    }
    model.fit();
    const p = model.predict(vec(300, 250));
    return Math.hypot(p.x - 300, p.y - 250);
  });
  note(err < 15, 'the model fits and predicts inside the browser', `${err.toFixed(1)} px off`);

  // The calibration overlay must build its own shadow DOM with no host markup.
  const overlay = await page.evaluate(async () => {
    const { CalibrationOverlay } = await import('./js/calibration-ui.js');
    const o = new CalibrationOverlay();
    o.mount();
    const host = document.querySelector('[data-eyetracker-overlay]');
    const hasDot = !!host?.shadowRoot?.querySelector('.dot');
    o.unmount();
    return { hasDot, cleaned: !document.querySelector('[data-eyetracker-overlay]') };
  });
  note(overlay.hasDot && overlay.cleaned, 'the calibration overlay mounts and tears down');

  note(pageErrors.length === 0, 'no uncaught page errors', pageErrors[0] ?? '');
  // MediaPipe logs its TFLite banner on the error channel; the service worker
  // and favicon are noisy too. Only genuine errors should count.
  const real = consoleErrors.filter(
    (t) => !/favicon|sw\.js|ServiceWorker|TensorFlow Lite|XNNPACK|^INFO:/i.test(t)
  );
  note(real.length === 0, 'no console errors', real.slice(0, 2).join(' | '));
} catch (err) {
  note(false, 'smoke run', err.message);
  await page.screenshot({ path: 'test/e2e/screenshot-failure.png' }).catch(() => {});
} finally {
  await browser.close();
}

console.log(failures.length ? `\n${failures.length} check(s) failed` : '\nall checks passed');
process.exit(failures.length ? 1 : 0);
