import test from 'node:test';
import assert from 'node:assert/strict';

import { FixationDetector } from '../js/fixation.js';

/** Feed `ms` worth of 60 Hz samples jittering around (x, y). */
function dwell(det, x, y, ms, t0, jitter = 2) {
  const out = [];
  for (let t = t0; t < t0 + ms; t += 1000 / 60) {
    const f = det.add(x + (Math.random() - 0.5) * jitter, y + (Math.random() - 0.5) * jitter, t);
    if (f) out.push(f);
  }
  return out;
}

test('a steady dwell followed by a jump yields one fixation', () => {
  const det = new FixationDetector();
  const ended = [];
  ended.push(...dwell(det, 400, 300, 500, 0));
  assert.equal(ended.length, 0, 'nothing closes while the eye is still');

  // A 600 px jump in one 16 ms frame is ~37500 px/s: unambiguously a saccade.
  const f = det.add(1000, 300, 520);
  assert.ok(f, 'the saccade closes the fixation');
  assert.ok(Math.abs(f.x - 400) < 5, `centroid x=${f.x}`);
  assert.ok(f.duration >= 480, `duration=${f.duration}`);
  assert.ok(f.samples > 25, `samples=${f.samples}`);
});

test('micro-dwells shorter than the minimum are discarded', () => {
  const det = new FixationDetector({ minDurationMs: 200 });
  dwell(det, 100, 100, 80, 0);
  assert.equal(det.add(900, 900, 100), null);
});

test('a gap in the stream ends the fixation', () => {
  const det = new FixationDetector({ maxGapMs: 200 });
  dwell(det, 200, 200, 400, 0);
  const f = det.add(200, 200, 1000); // 600 ms later: blink or lost face
  assert.ok(f, 'the gap closes it even though the position did not move');
  assert.ok(f.duration >= 380);
});

test('two dwells separated by a saccade give two fixations', () => {
  const det = new FixationDetector();
  const ended = [];
  ended.push(...dwell(det, 100, 100, 400, 0));
  ended.push(...dwell(det, 800, 600, 400, 420));
  const last = det.flush();
  if (last) ended.push(last);
  assert.equal(ended.length, 2, `got ${ended.length}`);
  assert.ok(Math.abs(ended[0].x - 100) < 5);
  assert.ok(Math.abs(ended[1].x - 800) < 5);
});

test('slow drift within the threshold stays a single fixation', () => {
  const det = new FixationDetector({ velocityThreshold: 600 });
  let ended = 0;
  // 120 px/s drift - well under the threshold.
  for (let i = 0; i < 60; i++) {
    if (det.add(300 + i * 2, 300, i * (1000 / 60))) ended++;
  }
  assert.equal(ended, 0);
  const f = det.flush();
  assert.ok(f && f.samples === 60, JSON.stringify(f));
});
