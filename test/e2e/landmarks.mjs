// Verify the landmark index map against MediaPipe's real output on a real face.
//
// Everything in js/features.js rests on a table of indices: 468 and 473 are
// the iris centres, 33/133 and 263/362 the eye corners, and so on. If any of
// those is wrong the tracker is wrong in a way no synthetic test can catch,
// because the synthetic faces are built from the same table. So run the actual
// model over an actual photograph and check the geometry makes sense.
//
// Usage: node test/e2e/landmarks.mjs [baseUrl]

import { chromium } from 'playwright';
import { mkdirSync, existsSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, '../..');
const FIXTURE = join(HERE, 'fixtures', 'portrait.jpg');
// MediaPipe's own test portrait: the same image their face-landmarker examples
// use, so it is a fair check of the published index map.
const FIXTURE_URL = 'https://storage.googleapis.com/mediapipe-assets/portrait.jpg';

const BASE = process.argv[2] ?? 'http://localhost:8099';
const failures = [];
const note = (ok, label, detail = '') => {
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}${detail ? ` — ${detail}` : ''}`);
  if (!ok) failures.push(label);
};

if (!existsSync(FIXTURE)) {
  console.log('downloading the test portrait ...');
  mkdirSync(dirname(FIXTURE), { recursive: true });
  const res = await fetch(FIXTURE_URL);
  if (!res.ok) {
    console.log(`FAIL could not fetch the fixture: HTTP ${res.status}`);
    process.exit(1);
  }
  writeFileSync(FIXTURE, Buffer.from(await res.arrayBuffer()));
}

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

  const r = await page.evaluate(async (fixturePath) => {
    const [{ resolveAssets }, { buildFeatures, IDX }] = await Promise.all([
      import('./js/config.js'),
      import('./js/features.js'),
    ]);
    const assets = resolveAssets();
    const { FaceLandmarker, FilesetResolver } = await import(assets.bundle);
    const fileset = await FilesetResolver.forVisionTasks(assets.wasm);
    const landmarker = await FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: assets.model, delegate: 'GPU' },
      runningMode: 'IMAGE',
      numFaces: 1,
    });

    const img = new Image();
    img.src = fixturePath;
    await img.decode();

    const result = landmarker.detect(img);
    const lm = result.faceLandmarks?.[0];
    if (!lm) return { error: 'no face detected in the fixture' };

    const aspect = img.naturalWidth / img.naturalHeight;
    const px = (i) => ({ x: lm[i].x * aspect, y: lm[i].y });
    const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

    const eye = (e) => {
      const outer = px(e.outer);
      const inner = px(e.inner);
      const upper = px(e.upper);
      const lower = px(e.lower);
      const iris = px(e.iris);
      const lo = Math.min(outer.x, inner.x);
      const hi = Math.max(outer.x, inner.x);
      return {
        width: dist(outer, inner),
        irisInsideHorizontally: iris.x > lo && iris.x < hi,
        // Lids are only ~1/3 as far apart as the corners, so allow the iris to
        // sit slightly outside the lid landmarks themselves.
        irisInsideVertically:
          iris.y > Math.min(upper.y, lower.y) - dist(upper, lower) &&
          iris.y < Math.max(upper.y, lower.y) + dist(upper, lower),
        ear: dist(upper, lower) / dist(outer, inner),
        upperAboveLower: upper.y < lower.y,
        irisRingRadius:
          [1, 2, 3, 4]
            .map((k) => dist(px(e.iris + k), iris))
            .reduce((a, b) => a + b, 0) / 4,
      };
    };

    const A = eye(IDX.eyeA);
    const B = eye(IDX.eyeB);
    const f = buildFeatures(lm, aspect);

    return {
      error: null,
      count: lm.length,
      // Landmark 33 is the outer corner of the image-left eye, 263 of the
      // image-right one, so this ordering must hold on a frontal face.
      eyeAOuterIsLeftmost: px(IDX.eyeA.outer).x < px(IDX.eyeA.inner).x,
      eyeBOuterIsRightmost: px(IDX.eyeB.outer).x > px(IDX.eyeB.inner).x,
      eyesAreSideBySide: px(IDX.eyeA.iris).x < px(IDX.eyeB.iris).x,
      noseBelowEyes: px(IDX.nose).y > px(IDX.eyeA.outer).y,
      chinBelowNose: px(IDX.chin).y > px(IDX.nose).y,
      browAboveEyes: px(IDX.brow).y < px(IDX.eyeA.outer).y,
      A,
      B,
      irisSeparationRatio:
        dist(px(IDX.eyeA.iris), px(IDX.eyeB.iris)) /
        dist(px(IDX.eyeA.outer), px(IDX.eyeB.outer)),
      features: { finite: f.vec.every(Number.isFinite), ear: f.ear, mx: f.vec[4], my: f.vec[5] },
    };
  }, '/test/e2e/fixtures/portrait.jpg');

  if (r.error) {
    note(false, 'landmark run', r.error);
  } else {
    note(r.count === 478, 'the model returns the iris-refined 478-point mesh', `${r.count}`);
    note(r.eyeAOuterIsLeftmost, 'IDX.eyeA corners are ordered outer -> inner');
    note(r.eyeBOuterIsRightmost, 'IDX.eyeB corners are ordered outer -> inner');
    note(r.eyesAreSideBySide, 'the two iris centres sit on opposite sides');
    note(r.A.upperAboveLower && r.B.upperAboveLower, 'lid landmarks are ordered upper/lower');
    note(r.noseBelowEyes && r.chinBelowNose && r.browAboveEyes, 'nose, chin and brow are where the map says');

    note(
      r.A.irisInsideHorizontally && r.B.irisInsideHorizontally,
      'each iris centre lies between its own eye corners'
    );
    note(
      r.A.irisInsideVertically && r.B.irisInsideVertically,
      'each iris centre lies within its own lid opening'
    );
    note(
      r.A.ear > 0.15 && r.A.ear < 0.6 && r.B.ear > 0.15 && r.B.ear < 0.6,
      'open eyes give a plausible eye-aspect ratio',
      `${r.A.ear.toFixed(2)} / ${r.B.ear.toFixed(2)}`
    );
    note(
      r.A.irisRingRadius > 0.05 * r.A.width && r.A.irisRingRadius < 0.6 * r.A.width,
      'the iris ring landmarks surround the centre at a sane radius',
      `${(r.A.irisRingRadius / r.A.width).toFixed(2)} of eye width`
    );
    note(
      r.irisSeparationRatio > 0.4 && r.irisSeparationRatio < 0.8,
      'iris separation is a plausible fraction of the corner-to-corner span',
      r.irisSeparationRatio.toFixed(2)
    );
    note(r.features.finite, 'buildFeatures produces finite values on a real face');
    note(
      Math.abs(r.features.mx) < 0.4 && Math.abs(r.features.my) < 0.4,
      'iris offsets are near zero for a roughly forward gaze',
      `mx=${r.features.mx.toFixed(3)} my=${r.features.my.toFixed(3)}`
    );
  }

  note(pageErrors.length === 0, 'no uncaught page errors', pageErrors[0] ?? '');
} catch (err) {
  note(false, 'landmark run', err.message);
} finally {
  await browser.close();
}

console.log(failures.length ? `\n${failures.length} check(s) failed` : '\nall checks passed');
process.exit(failures.length ? 1 : 0);
