// End-to-end calibration run in a real browser, driven by a virtual user.
//
// Instead of a camera we inject synthetic landmarks whose iris position is a
// known function of "where this user is looking", and we point that gaze at
// whatever the calibration overlay is currently showing. Playwright clicks the
// targets. That exercises the parts a unit test cannot reach: the shadow-DOM
// overlay, the click/collect state machine, sample grouping, and the fit -
// through the same code path a real user takes.
//
// Usage: node test/e2e/calibration.mjs [baseUrl]

import { chromium } from 'playwright';

const BASE = process.argv[2] ?? 'http://localhost:8099';
const failures = [];
const note = (ok, label, detail = '') => {
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}${detail ? ` — ${detail}` : ''}`);
  if (!ok) failures.push(label);
};

const browser = await chromium.launch();
const context = await browser.newContext({ viewport: { width: 1280, height: 800 } });
await context.addInitScript(() => {
  globalThis.EYETRACKER_ASSETS = '/vendor';
});
const page = await context.newPage();
const pageErrors = [];
page.on('pageerror', (e) => pageErrors.push(String(e)));

try {
  await page.goto(BASE, { waitUntil: 'load', timeout: 30000 });

  // Build a tracker with no camera behind it and start feeding virtual frames.
  await page.evaluate(async () => {
    const [{ EyeTracker }, { IDX }] = await Promise.all([
      import('./js/eyetracker.js'),
      import('./js/features.js'),
    ]);

    const et = new EyeTracker({ smoothing: 0 });
    et.tracker.running = true; // no camera; we drive _onFrame ourselves
    globalThis.__et = et;
    globalThis.__gazeEvents = [];
    et.on('gaze', (g) => globalThis.__gazeEvents.push(g));

    // Same forward model the unit tests use: the eye rotates to look at the
    // target, with a little noise and head movement on top.
    const face = (gx, gy, yaw) => {
      const lm = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
      const put = (i, x, y) => (lm[i] = { x, y, z: 0 });
      const n = () => (Math.random() - 0.5) * 0.0008;
      const ox = 0.013 * (gx - 0.8 * yaw) + n();
      const oy = 0.009 * gy + n();
      for (const [eye, cx] of [[IDX.eyeA, 0.43], [IDX.eyeB, 0.57]]) {
        put(eye.outer, cx + (cx < 0.5 ? -0.03 : 0.03), 0.45);
        put(eye.inner, cx + (cx < 0.5 ? 0.03 : -0.03), 0.45);
        put(eye.upper, cx, 0.435);
        put(eye.lower, cx, 0.465);
        put(eye.iris, cx + ox, 0.45 + oy);
      }
      put(IDX.nose, 0.5 + 0.06 * yaw, 0.55);
      put(IDX.chin, 0.5 + 0.03 * yaw, 0.75);
      put(IDX.brow, 0.5 + 0.02 * yaw, 0.28);
      return lm;
    };

    /** Where the virtual user is looking: the calibration dot, if one is up. */
    globalThis.__lookAt = { x: innerWidth / 2, y: innerHeight / 2 };
    const currentTarget = () => {
      const host = document.querySelector('[data-eyetracker-overlay]');
      const dot = host?.shadowRoot?.querySelector('.dot');
      if (!dot) return globalThis.__lookAt;
      const r = dot.getBoundingClientRect();
      return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
    };

    const tick = () => {
      const t = currentTarget();
      const yaw = (Math.random() - 0.5) * 0.12;
      et._onFrame({
        landmarks: face((t.x / innerWidth) * 2 - 1, (t.y / innerHeight) * 2 - 1, yaw),
        timestamp: performance.now(),
      });
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);

    globalThis.collectingNow = () =>
      document
        .querySelector('[data-eyetracker-overlay]')
        ?.shadowRoot?.querySelector('.dot')
        ?.classList.contains('collecting') ?? false;
  });
  note(true, 'virtual user is feeding frames');

  // Kick off calibration; it resolves only once every target is done.
  // Kept as a floating promise while Playwright drives the clicks; give it a
  // catch so a failure later cannot surface as an unhandled rejection that
  // hides the real error.
  const calibration = page.evaluate(() =>
    globalThis.__et.calibrate({ mode: 'click' }).then((r) => ({
      cv: r?.cv ?? null,
      lambda: r?.lambda ?? null,
      groups: r?.groups ?? 0,
      samples: r?.samples ?? 0,
    }))
  ).catch((err) => ({ error: err.message, groups: 0, samples: 0, cv: null, lambda: null }));

  const dot = page.locator('[data-eyetracker-overlay] .dot');
  for (let i = 0; i < 9; i++) {
    await dot.waitFor({ state: 'visible', timeout: 15000 });
    // The dot is marked while it records; wait for it to go idle again.
    await page.waitForFunction(() => !collectingNow(), null, { timeout: 15000, polling: 50 });
    // No force: the dot glides to its next position with a CSS transition, and
    // clicking mid-transition lands on empty overlay. Let Playwright wait for
    // the element to be stable first.
    await dot.click({ timeout: 15000 });
    await page.waitForFunction(() => collectingNow(), null, { timeout: 15000, polling: 50 });
  }

  const report = await calibration;
  note(report.groups === 9, 'all nine targets were recorded', `${report.groups} groups`);
  note(report.samples > 200, 'enough samples were collected', `${report.samples}`);
  note(report.lambda > 0, 'cross-validation selected a ridge strength', `λ=${report.lambda}`);
  note(report.cv !== null && report.cv < 120, 'held-out error is sane', `${report.cv?.toFixed(0)} px`);

  note(
    !(await page.locator('[data-eyetracker-overlay]').count()),
    'the overlay tore itself down afterwards'
  );

  // Now that a model exists, the live loop should be producing gaze events
  // that land on wherever the virtual user is told to look.
  const live = await page.evaluate(async () => {
    const target = { x: 300, y: 600 };
    globalThis.__lookAt = target;
    globalThis.__gazeEvents.length = 0;
    await new Promise((r) => setTimeout(r, 700));
    const events = globalThis.__gazeEvents.slice(-20);
    if (events.length === 0) return { error: 'no gaze events' };
    const mx = events.reduce((a, e) => a + e.x, 0) / events.length;
    const my = events.reduce((a, e) => a + e.y, 0) / events.length;
    return {
      error: null,
      distance: Math.hypot(mx - target.x, my - target.y),
      count: events.length,
      quality: events.at(-1).quality?.ok,
      blinkThreshold: globalThis.__et.blinkThreshold,
    };
  });

  note(!live.error, 'the live loop emits gaze events', live.error ?? `${live.count} events`);
  note(live.distance < 100, 'live gaze lands on the target', `${live.distance?.toFixed(0)} px off`);
  note(live.quality === true, 'pose quality reads as in-range for a steady head');
  note(
    live.blinkThreshold > 0.1 && live.blinkThreshold <= 0.25,
    'a per-user blink threshold was learned',
    live.blinkThreshold?.toFixed(3)
  );

  // Smooth pursuit needs no clicks: the virtual user simply keeps looking at
  // wherever the dot is, which is exactly what the lag correction assumes.
  //
  // The speed here is ~5x the default to keep the test short, which also makes
  // it much harder than the real thing: at 2200 px/s the 120 ms lag correction
  // spans 260 px, and only ~140 samples land. The accuracy bound below is
  // loose for that reason - it checks the path works, not how good it gets.
  const pursuit = await page.evaluate(async () => {
    const r = await globalThis.__et.calibrate({ mode: 'pursuit', speed: 2200 });
    if (!r) return { cancelled: true };
    const target = { x: 900, y: 250 };
    globalThis.__lookAt = target;
    globalThis.__gazeEvents.length = 0;
    await new Promise((res) => setTimeout(res, 700));
    const events = globalThis.__gazeEvents.slice(-20);
    const mx = events.reduce((a, e) => a + e.x, 0) / events.length;
    const my = events.reduce((a, e) => a + e.y, 0) / events.length;
    return {
      groups: r.groups,
      samples: r.samples,
      cv: r.cv,
      distance: Math.hypot(mx - target.x, my - target.y),
    };
  });

  note(!pursuit.cancelled && pursuit.groups >= 5, 'pursuit calibration produced path groups', `${pursuit.groups} groups`);
  note(pursuit.samples > 100, 'pursuit collected samples continuously', `${pursuit.samples}`);
  note(pursuit.distance < 150, 'pursuit-only model lands on a held-out point', `${pursuit.distance?.toFixed(0)} px off`);

  note(pageErrors.length === 0, 'no uncaught page errors', pageErrors[0] ?? '');
} catch (err) {
  note(false, 'calibration run', err.message);
  await page.screenshot({ path: 'test/e2e/screenshot-calibration-failure.png' }).catch(() => {});
} finally {
  await browser.close();
}

console.log(failures.length ? `\n${failures.length} check(s) failed` : '\nall checks passed');
process.exit(failures.length ? 1 : 0);
