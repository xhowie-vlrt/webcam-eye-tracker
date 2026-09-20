// The gaze model: feature vectors in, screen coordinates out.

import { ridgeFit, ridgePredict } from './math.js';
import { FEATURE_DIM } from './features.js';

const STORAGE_KEY = 'webcam-eye-tracker:model:v1';

export class GazeModel {
  constructor() {
    /** @type {{X:number[][], Y:number[][]}} */
    this.data = { X: [], Y: [] };
    this.model = null;
    this.lambda = 0.05;
    /** Residual stats from the last fit, in CSS pixels. */
    this.trainError = null;
  }

  get ready() {
    return this.model !== null;
  }

  get sampleCount() {
    return this.data.X.length;
  }

  /** Add one observation: feature vector -> known screen point. */
  addSample(vec, x, y) {
    if (vec.length !== FEATURE_DIM) {
      throw new Error(`expected ${FEATURE_DIM} features, got ${vec.length}`);
    }
    this.data.X.push(vec);
    this.data.Y.push([x, y]);
  }

  clear() {
    this.data = { X: [], Y: [] };
    this.model = null;
    this.trainError = null;
  }

  fit() {
    const { X, Y } = this.data;
    // Ridge keeps this solvable well below the theoretical minimum, but a
    // handful of samples still produces a useless model.
    if (X.length < 20) throw new Error('need at least 20 samples to fit');
    this.model = ridgeFit(X, Y, this.lambda);
    this.trainError = this.errorOn(X, Y);
    return this.trainError;
  }

  predict(vec) {
    if (!this.model) return null;
    const [x, y] = ridgePredict(this.model, vec);
    return { x, y };
  }

  errorOn(X, Y) {
    let sum = 0;
    let max = 0;
    for (let i = 0; i < X.length; i++) {
      const p = ridgePredict(this.model, X[i]);
      const e = Math.hypot(p[0] - Y[i][0], p[1] - Y[i][1]);
      sum += e;
      if (e > max) max = e;
    }
    return { mean: sum / X.length, max, n: X.length };
  }

  save() {
    if (!this.model) return false;
    try {
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({
          model: this.model,
          trainError: this.trainError,
          screen: { w: window.innerWidth, h: window.innerHeight },
          savedAt: Date.now(),
        })
      );
      return true;
    } catch {
      return false;
    }
  }

  load() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return null;
      const parsed = JSON.parse(raw);
      if (!parsed?.model?.W) return null;
      this.model = parsed.model;
      this.trainError = parsed.trainError ?? null;
      return parsed;
    } catch {
      return null;
    }
  }

  static forget() {
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {
      /* private mode - nothing to do */
    }
  }
}

/**
 * Rough angular error, assuming a typical desktop setup. Screen-pixel error is
 * what you can measure; degrees is what eye-tracking papers quote, so show both
 * and be honest that the conversion is an assumption.
 */
export function pixelsToDegrees(pixels, viewingDistanceCm = 60) {
  const cmPerPx = 2.54 / 96; // CSS px are defined as 1/96 inch
  return (Math.atan((pixels * cmPerPx) / viewingDistanceCm) * 180) / Math.PI;
}
