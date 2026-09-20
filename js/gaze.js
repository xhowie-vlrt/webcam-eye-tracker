// The gaze model: feature vectors in, screen coordinates out.
//
// Three things separate this from a plain least-squares fit, and all three
// matter in practice:
//
//   1. The ridge strength is chosen by leave-one-group-out cross-validation,
//      where a "group" is one calibration target. Samples from a single
//      fixation are almost identical, so random k-fold would leak and pick a
//      far too small lambda.
//   2. The fit is repeated with Cauchy weights (IRLS). A single blink or
//      mis-detected iris during calibration otherwise drags the whole map.
//   3. Samples can be added online at a lower weight, which is how drift
//      correction (assume the user looks at what they click) feeds back in.

import { ridgeFit, ridgePredict } from './math.js';
import { FEATURE_DIM } from './features.js';

const STORAGE_KEY = 'webcam-eye-tracker:model:v2';
const LAMBDAS = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1];
const IRLS_ROUNDS = 3;
const MIN_SAMPLES = 20;

export class GazeModel {
  /**
   * @param {object} [opts]
   * @param {number} [opts.maxSamples] ring-buffer cap, so online corrections
   *   cannot grow the training set without bound
   */
  constructor({ maxSamples = 6000 } = {}) {
    this.maxSamples = maxSamples;
    this.samples = []; // { vec, x, y, group, weight }
    this.model = null;
    this.lambda = null;
    this.trainError = null;
    this.cvError = null;
  }

  get ready() {
    return this.model !== null;
  }

  get sampleCount() {
    return this.samples.length;
  }

  /**
   * @param {number[]} vec feature vector
   * @param {number} x screen x of the known target
   * @param {number} y screen y
   * @param {object} [opts]
   * @param {number|string} [opts.group] fixation id; samples sharing one stay
   *   together during cross-validation
   * @param {number} [opts.weight] relative trust, 1 for explicit calibration
   */
  addSample(vec, x, y, { group = 0, weight = 1 } = {}) {
    if (vec.length !== FEATURE_DIM) {
      throw new Error(`expected ${FEATURE_DIM} features, got ${vec.length}`);
    }
    this.samples.push({ vec, x, y, group, weight });
    if (this.samples.length > this.maxSamples) {
      // Drop the oldest *online* sample first; calibration data is the anchor
      // and should outlive drift corrections.
      const i = this.samples.findIndex((s) => s.weight < 1);
      this.samples.splice(i === -1 ? 0 : i, 1);
    }
  }

  clear() {
    this.samples = [];
    this.model = null;
    this.lambda = null;
    this.trainError = null;
    this.cvError = null;
  }

  /** Remove everything added by drift correction, keeping the calibration. */
  clearOnlineSamples() {
    this.samples = this.samples.filter((s) => s.weight >= 1);
  }

  /**
   * @param {object} [opts]
   * @param {boolean} [opts.robust] run the IRLS reweighting rounds
   * @param {boolean} [opts.crossValidate] pick lambda by leave-one-group-out CV
   */
  fit({ robust = true, crossValidate = true } = {}) {
    const n = this.samples.length;
    if (n < MIN_SAMPLES) {
      throw new Error(`need at least ${MIN_SAMPLES} samples to fit`);
    }

    const groups = [...new Set(this.samples.map((s) => s.group))];
    let lambda = 1e-2;
    let cvError = null;

    if (crossValidate && groups.length >= 3) {
      let best = Infinity;
      for (const candidate of LAMBDAS) {
        const e = this._crossValidate(candidate, groups);
        if (e < best) {
          best = e;
          lambda = candidate;
        }
      }
      cvError = best;
    }

    this.lambda = lambda;
    this.cvError = cvError;
    this.model = this._fitWeighted(this.samples, lambda, robust);
    this.trainError = this.evaluate(this.samples);
    return { train: this.trainError, cv: cvError, lambda };
  }

  /** Refit in place, e.g. after drift-correction samples arrive. */
  refit() {
    if (this.samples.length < MIN_SAMPLES) return null;
    this.model = this._fitWeighted(this.samples, this.lambda ?? 1e-2, true);
    this.trainError = this.evaluate(this.samples);
    return this.trainError;
  }

  predict(vec) {
    if (!this.model) return null;
    const [x, y] = ridgePredict(this.model, vec);
    return { x, y };
  }

  /** Mean / 95th-percentile / max euclidean error over a set of samples. */
  evaluate(samples, model = this.model) {
    if (!model || samples.length === 0) return null;
    const errors = samples.map((s) => {
      const p = ridgePredict(model, s.vec);
      return Math.hypot(p[0] - s.x, p[1] - s.y);
    });
    errors.sort((a, b) => a - b);
    const sum = errors.reduce((a, b) => a + b, 0);
    return {
      mean: sum / errors.length,
      p95: errors[Math.min(errors.length - 1, Math.floor(errors.length * 0.95))],
      max: errors[errors.length - 1],
      n: errors.length,
    };
  }

  // -- internals ------------------------------------------------------------

  _fitWeighted(samples, lambda, robust) {
    const X = samples.map((s) => s.vec);
    const Y = samples.map((s) => [s.x, s.y]);
    let weights = samples.map((s) => s.weight);
    let model = ridgeFit(X, Y, lambda, weights);
    if (!robust) return model;

    for (let round = 0; round < IRLS_ROUNDS; round++) {
      const residuals = X.map((vec, i) => {
        const p = ridgePredict(model, vec);
        return Math.hypot(p[0] - Y[i][0], p[1] - Y[i][1]);
      });
      // Median absolute residual is a scale estimate that outliers cannot move.
      const sorted = [...residuals].sort((a, b) => a - b);
      const scale = Math.max(1e-6, sorted[Math.floor(sorted.length / 2)] * 1.4826);
      // Cauchy: down-weights smoothly, never hard-drops a sample.
      weights = samples.map((s, i) => {
        const r = residuals[i] / (2.385 * scale);
        return s.weight / (1 + r * r);
      });
      model = ridgeFit(X, Y, lambda, weights);
    }
    return model;
  }

  _crossValidate(lambda, groups) {
    let total = 0;
    let count = 0;
    for (const g of groups) {
      const train = this.samples.filter((s) => s.group !== g);
      const held = this.samples.filter((s) => s.group === g);
      if (train.length < MIN_SAMPLES || held.length === 0) continue;
      let model;
      try {
        model = this._fitWeighted(train, lambda, true);
      } catch {
        return Infinity; // this lambda cannot be solved for; never pick it
      }
      // Score the *mean residual* of the group, not the mean prediction
      // against one target: a pursuit group is a stretch of path where every
      // sample has its own target, so held[0] is not the group's position.
      // Averaging the residual keeps per-frame noise out of model selection
      // while staying correct for both discrete and moving targets.
      let ex = 0;
      let ey = 0;
      for (const s of held) {
        const p = ridgePredict(model, s.vec);
        ex += p[0] - s.x;
        ey += p[1] - s.y;
      }
      total += Math.hypot(ex / held.length, ey / held.length);
      count++;
    }
    return count === 0 ? Infinity : total / count;
  }

  // -- persistence ----------------------------------------------------------

  save() {
    if (!this.model) return false;
    try {
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({
          model: this.model,
          lambda: this.lambda,
          trainError: this.trainError,
          cvError: this.cvError,
          screen: { w: window.innerWidth, h: window.innerHeight },
          savedAt: Date.now(),
        })
      );
      return true;
    } catch {
      return false; // quota or private mode; not worth failing the session over
    }
  }

  load() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return null;
      const parsed = JSON.parse(raw);
      if (!parsed?.model?.W || parsed.model.mean?.length !== FEATURE_DIM) return null;
      this.model = parsed.model;
      this.lambda = parsed.lambda ?? null;
      this.trainError = parsed.trainError ?? null;
      this.cvError = parsed.cvError ?? null;
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
