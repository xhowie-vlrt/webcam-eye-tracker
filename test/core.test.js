// Unit tests for the parts that do not need a browser.
// Run with: npm test   (node --test)

import test from 'node:test';
import assert from 'node:assert/strict';

import { solve, ridgeFit, ridgePredict } from '../js/math.js';
import { OneEuroFilter } from '../js/filter.js';
import { buildFeatures, FEATURE_DIM, IDX } from '../js/features.js';

test('solve inverts a small system', () => {
  const A = [
    [2, 1, 0],
    [1, 3, 1],
    [0, 1, 2],
  ];
  const X = [[1], [2], [3]];
  const B = A.map((row) => [row.reduce((s, v, j) => s + v * X[j][0], 0)]);
  const got = solve(A, B);
  for (let i = 0; i < 3; i++) {
    assert.ok(Math.abs(got[i][0] - X[i][0]) < 1e-9, `row ${i}: ${got[i][0]}`);
  }
});

test('solve reports singular systems instead of returning garbage', () => {
  assert.throws(
    () => solve([[1, 2], [2, 4]], [[1], [2]]),
    /singular/
  );
});

test('ridge recovers a linear map when the penalty is tiny', () => {
  const truth = [
    [3, -1],
    [0.5, 2],
  ];
  const bias = [10, -4];
  const X = [];
  const Y = [];
  for (let i = 0; i < 200; i++) {
    const a = Math.sin(i * 0.7);
    const b = Math.cos(i * 0.31) * 2;
    X.push([a, b]);
    Y.push([
      bias[0] + truth[0][0] * a + truth[1][0] * b,
      bias[1] + truth[0][1] * a + truth[1][1] * b,
    ]);
  }
  const model = ridgeFit(X, Y, 1e-9);
  const p = ridgePredict(model, [0.4, -1.1]);
  assert.ok(Math.abs(p[0] - (bias[0] + 3 * 0.4 + 0.5 * -1.1)) < 1e-6, `x=${p[0]}`);
  assert.ok(Math.abs(p[1] - (bias[1] + -1 * 0.4 + 2 * -1.1)) < 1e-6, `y=${p[1]}`);
});

test('ridge survives a constant (zero-variance) feature column', () => {
  const X = Array.from({ length: 40 }, (_, i) => [i / 40, 1]);
  const Y = X.map(([a]) => [a * 2, 5]);
  const model = ridgeFit(X, Y, 1e-6);
  assert.ok(Number.isFinite(ridgePredict(model, [0.5, 1])[0]));
});

test('one euro filter settles on a constant signal', () => {
  const f = new OneEuroFilter({ minCutoff: 1, beta: 0.01 });
  let out = 0;
  for (let i = 0; i < 120; i++) out = f.filter(5, i / 60);
  assert.ok(Math.abs(out - 5) < 1e-3, `settled at ${out}`);
});

test('one euro filter reduces noise', () => {
  const f = new OneEuroFilter({ minCutoff: 0.5, beta: 0.005 });
  let noisy = 0;
  let smooth = 0;
  for (let i = 0; i < 300; i++) {
    const v = 10 + (i % 2 ? 1 : -1);
    noisy += Math.abs(v - 10);
    smooth += Math.abs(f.filter(v, i / 60) - 10);
  }
  assert.ok(smooth < noisy / 4, `smooth=${smooth} noisy=${noisy}`);
});

/** Minimal synthetic face: every landmark defined, key ones placed sensibly. */
function fakeFace({ irisShift = 0 } = {}) {
  const lm = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
  const put = (i, x, y) => {
    lm[i] = { x, y, z: 0 };
  };
  put(IDX.eyeA.outer, 0.40, 0.45);
  put(IDX.eyeA.inner, 0.46, 0.45);
  put(IDX.eyeA.upper, 0.43, 0.435);
  put(IDX.eyeA.lower, 0.43, 0.465);
  put(IDX.eyeA.iris, 0.43 + irisShift, 0.45);
  put(IDX.eyeB.outer, 0.60, 0.45);
  put(IDX.eyeB.inner, 0.54, 0.45);
  put(IDX.eyeB.upper, 0.57, 0.435);
  put(IDX.eyeB.lower, 0.57, 0.465);
  put(IDX.eyeB.iris, 0.57 + irisShift, 0.45);
  put(IDX.nose, 0.50, 0.55);
  put(IDX.chin, 0.50, 0.75);
  put(IDX.brow, 0.50, 0.28);
  return lm;
}

test('features are finite and the expected length', () => {
  const f = buildFeatures(fakeFace(), 16 / 9);
  assert.equal(f.vec.length, FEATURE_DIM);
  assert.ok(f.vec.every(Number.isFinite), 'every feature is finite');
  assert.ok(f.ear > 0 && f.ear < 1, `ear=${f.ear}`);
});

test('moving the iris moves the horizontal feature, monotonically', () => {
  const left = buildFeatures(fakeFace({ irisShift: -0.01 }), 16 / 9);
  const centre = buildFeatures(fakeFace({ irisShift: 0 }), 16 / 9);
  const right = buildFeatures(fakeFace({ irisShift: 0.01 }), 16 / 9);
  // index 4 is the mean horizontal iris offset
  assert.ok(left.vec[4] < centre.vec[4], 'left < centre');
  assert.ok(centre.vec[4] < right.vec[4], 'centre < right');
  assert.ok(Math.abs(centre.vec[4]) < 1e-9, 'centred iris is ~0');
});

test('closed lids drop the eye aspect ratio', () => {
  const open = buildFeatures(fakeFace(), 16 / 9);
  const lm = fakeFace();
  lm[IDX.eyeA.upper].y = 0.4495;
  lm[IDX.eyeA.lower].y = 0.4505;
  lm[IDX.eyeB.upper].y = 0.4495;
  lm[IDX.eyeB.lower].y = 0.4505;
  const closed = buildFeatures(lm, 16 / 9);
  assert.ok(closed.ear < 0.17, `closed ear=${closed.ear}`);
  assert.ok(open.ear > closed.ear * 3, 'open is clearly larger');
});
