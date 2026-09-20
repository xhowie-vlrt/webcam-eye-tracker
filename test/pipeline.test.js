// End-to-end numeric check of the calibrate -> predict pipeline, using a
// synthetic face whose iris position is a known function of gaze and head
// pose. If the regressor cannot recover that mapping here, it has no chance
// in front of a real camera.

import test from 'node:test';
import assert from 'node:assert/strict';

import { buildFeatures, IDX } from '../js/features.js';
import { GazeModel } from '../js/gaze.js';

const SCREEN = { w: 1440, h: 900 };
const ASPECT = 16 / 9;

// Deterministic PRNG so a failure is always reproducible.
function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

/**
 * Forward model. The eye rotates by (gaze - headYaw), so the iris offset and
 * the head-pose proxy are correlated - exactly the confound the extra features
 * exist to resolve.
 */
function synthFace({ gx, gy, yaw = 0, dx = 0, dy = 0, noise = 0, rand = Math.random }) {
  const jitter = () => (rand() - 0.5) * 2 * noise;
  const lm = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
  const put = (i, x, y) => {
    lm[i] = { x: x + dx, y: y + dy, z: 0 };
  };

  const eyeInHeadX = gx - 0.8 * yaw;
  const eyeInHeadY = gy;
  const ox = 0.013 * eyeInHeadX + jitter();
  const oy = 0.009 * eyeInHeadY + jitter();

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
}

function sample(target, rand, noise) {
  return buildFeatures(
    synthFace({
      gx: (target.x / SCREEN.w) * 2 - 1,
      gy: (target.y / SCREEN.h) * 2 - 1,
      yaw: (rand() - 0.5) * 0.3,
      dx: (rand() - 0.5) * 0.02,
      dy: (rand() - 0.5) * 0.02,
      noise,
      rand,
    }),
    ASPECT
  ).vec;
}

function grid(cols, rows, inset = 0.08) {
  const pts = [];
  const span = 1 - 2 * inset;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      pts.push({
        x: (inset + (c / (cols - 1)) * span) * SCREEN.w,
        y: (inset + (r / (rows - 1)) * span) * SCREEN.h,
      });
    }
  }
  return pts;
}

function train({ noise, seed = 7, outlierRate = 0, fitOpts = {} }) {
  const rand = rng(seed);
  const model = new GazeModel();
  grid(3, 3).forEach((p, group) => {
    for (let i = 0; i < 25; i++) {
      // An outlier stands in for a blink or a mis-detected iris: the target is
      // right, the features belong to a completely different gaze direction.
      const bad = rand() < outlierRate;
      const src = bad
        ? { x: rand() * SCREEN.w, y: rand() * SCREEN.h }
        : p;
      model.addSample(sample(src, rand, noise), p.x, p.y, { group });
    }
  });
  model.fit(fitOpts);
  return { model, rand };
}

function measure(model, rand, noise) {

  // Validate on points that were never calibration targets.
  const held = [
    { x: 0.25 * SCREEN.w, y: 0.25 * SCREEN.h },
    { x: 0.75 * SCREEN.w, y: 0.3 * SCREEN.h },
    { x: 0.5 * SCREEN.w, y: 0.5 * SCREEN.h },
    { x: 0.3 * SCREEN.w, y: 0.8 * SCREEN.h },
    { x: 0.8 * SCREEN.w, y: 0.7 * SCREEN.h },
  ];
  let total = 0;
  for (const p of held) {
    let sx = 0;
    let sy = 0;
    const n = 30;
    for (let i = 0; i < n; i++) {
      const q = model.predict(sample(p, rand, noise));
      sx += q.x;
      sy += q.y;
    }
    total += Math.hypot(sx / n - p.x, sy / n - p.y);
  }
  return total / held.length;
}

function trainAndMeasure(opts) {
  const { model, rand } = train(opts);
  return measure(model, rand, opts.noise);
}

test('9-point calibration recovers a noiseless mapping almost exactly', () => {
  const err = trainAndMeasure({ noise: 0 });
  assert.ok(err < 10, `mean held-out error ${err.toFixed(1)} px`);
});

test('calibration still generalises with landmark noise', () => {
  // 0.0006 of the frame width is roughly the jitter a 720p landmarker shows.
  const err = trainAndMeasure({ noise: 0.0006 });
  assert.ok(err < 120, `mean held-out error ${err.toFixed(1)} px`);
});

test('cross-validation picks a lambda and reports a held-out error', () => {
  const { model } = train({ noise: 0.0006 });
  assert.ok(model.lambda > 0, `lambda=${model.lambda}`);
  assert.ok(Number.isFinite(model.cvError), `cvError=${model.cvError}`);
  // The CV number is a leave-one-target-out estimate, so it should be in the
  // same ballpark as the held-out error, not the (optimistic) training error.
  assert.ok(model.cvError < 200, `cvError=${model.cvError.toFixed(1)} px`);
});

test('robust fitting beats plain least squares when calibration has outliers', () => {
  const opts = { noise: 0.0006, outlierRate: 0.15, seed: 11 };
  const plain = trainAndMeasure({ ...opts, fitOpts: { robust: false } });
  const robust = trainAndMeasure({ ...opts, fitOpts: { robust: true } });
  assert.ok(
    robust < plain,
    `robust=${robust.toFixed(1)} px should beat plain=${plain.toFixed(1)} px`
  );
});

test('evaluate reports mean, p95 and max', () => {
  const { model } = train({ noise: 0.0006 });
  const e = model.evaluate(model.samples);
  assert.ok(e.mean <= e.p95 && e.p95 <= e.max, JSON.stringify(e));
  assert.equal(e.n, 9 * 25);
});

test('online samples are dropped before calibration samples', () => {
  const m = new GazeModel({ maxSamples: 30 });
  const rand = rng(5);
  const vec = () => sample({ x: 100, y: 100 }, rand, 0);
  for (let i = 0; i < 25; i++) m.addSample(vec(), 100, 100, { group: 0 });
  for (let i = 0; i < 10; i++) m.addSample(vec(), 200, 200, { group: 1, weight: 0.3 });
  assert.equal(m.sampleCount, 30);
  assert.equal(m.samples.filter((s) => s.weight >= 1).length, 25);
});

test('an unfitted model predicts nothing rather than guessing', () => {
  assert.equal(new GazeModel().predict([]), null);
});

test('fitting refuses to run on a handful of samples', () => {
  const m = new GazeModel();
  const rand = rng(3);
  for (let i = 0; i < 5; i++) m.addSample(sample({ x: 100, y: 100 }, rand, 0), 100, 100);
  assert.throws(() => m.fit(), /at least 20 samples/);
});
