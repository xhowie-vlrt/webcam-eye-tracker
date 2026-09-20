import test from 'node:test';
import assert from 'node:assert/strict';

import { gridPoints, validationPoints, shuffle } from '../js/points.js';
import { pathLength, pointAt, serpentine } from '../js/calibration-ui.js';

const SIZE = { width: 1000, height: 800 };

test('grid points cover the viewport without touching the edges', () => {
  const pts = gridPoints(3, 3, { ...SIZE, inset: 0.07 });
  assert.equal(pts.length, 9);
  for (const p of pts) {
    assert.ok(p.x >= 70 && p.x <= 930, `x=${p.x}`);
    assert.ok(p.y >= 56 && p.y <= 744, `y=${p.y}`);
  }
  // Corners and centre are present.
  const near = (p, x, y) =>
    assert.ok(Math.hypot(p.x - x, p.y - y) < 1e-6, `${p.x},${p.y} != ${x},${y}`);
  near(pts[0], 70, 56);
  near(pts[4], 500, 400);
});

test('a single-column grid centres horizontally', () => {
  const pts = gridPoints(1, 3, SIZE);
  assert.ok(pts.every((p) => p.x === 500));
});

test('validation points avoid the calibration grid nodes', () => {
  for (const cols of [3, 4]) {
    const grid = gridPoints(cols, cols, SIZE);
    for (const v of validationPoints(SIZE)) {
      const nearest = Math.min(...grid.map((g) => Math.hypot(g.x - v.x, g.y - v.y)));
      assert.ok(
        nearest > 40,
        `validation point ${v.x},${v.y} sits on a ${cols}x${cols} grid node`
      );
    }
  }
});

test('shuffle keeps every element exactly once', () => {
  const input = Array.from({ length: 50 }, (_, i) => i);
  let seed = 1;
  const rand = () => {
    seed = (seed * 1103515245 + 12345) % 2147483648;
    return seed / 2147483648;
  };
  const out = shuffle(input, rand);
  assert.equal(out.length, input.length);
  assert.deepEqual([...out].sort((a, b) => a - b), input);
  assert.notDeepEqual(out, input);
});

test('serpentine alternates direction and spans the viewport', () => {
  const path = serpentine(1000, 800, { rows: 4, inset: 0.07 });
  assert.equal(path.length, 8);
  assert.ok(path[0].x < path[1].x, 'first row runs left to right');
  assert.ok(path[2].x > path[3].x, 'second row runs right to left');
  const ys = [...new Set(path.map((p) => p.y))];
  assert.equal(ys.length, 4);
  assert.ok(Math.min(...ys) >= 56 && Math.max(...ys) <= 744);
});

test('pointAt walks the polyline at the requested distance', () => {
  const path = [
    { x: 0, y: 0 },
    { x: 100, y: 0 },
    { x: 100, y: 100 },
  ];
  assert.equal(pathLength(path), 200);

  const a = pointAt(path, 50);
  assert.deepEqual({ x: a.x, y: a.y, segment: a.segment }, { x: 50, y: 0, segment: 0 });

  const b = pointAt(path, 150);
  assert.deepEqual({ x: b.x, y: b.y, segment: b.segment }, { x: 100, y: 50, segment: 1 });

  // Past the end it clamps rather than extrapolating off-screen.
  const c = pointAt(path, 500);
  assert.equal(c.x, 100);
  assert.equal(c.y, 100);
});

test('pointAt tolerates a zero-length segment', () => {
  const path = [
    { x: 10, y: 10 },
    { x: 10, y: 10 },
    { x: 50, y: 10 },
  ];
  const p = pointAt(path, 20);
  assert.equal(p.y, 10);
  assert.ok(p.x >= 10 && p.x <= 50, `x=${p.x}`);
});
