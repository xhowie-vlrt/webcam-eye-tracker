// 1-Euro filter (Casiez et al., CHI 2012).
// Low lag when the gaze moves fast, heavy smoothing when it is still - which is
// exactly the trade-off a jittery webcam signal needs.

class LowPass {
  constructor() {
    this.y = null;
  }
  filter(value, alpha) {
    this.y = this.y === null ? value : alpha * value + (1 - alpha) * this.y;
    return this.y;
  }
  reset() {
    this.y = null;
  }
}

function alphaFor(cutoff, dt) {
  const tau = 1 / (2 * Math.PI * cutoff);
  return 1 / (1 + tau / dt);
}

export class OneEuroFilter {
  /**
   * @param {object} opts
   * @param {number} opts.minCutoff lower = smoother when still
   * @param {number} opts.beta      higher = less lag when moving fast
   * @param {number} opts.dCutoff   cutoff of the speed estimator
   */
  constructor({ minCutoff = 1.0, beta = 0.02, dCutoff = 1.0 } = {}) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.x = new LowPass();
    this.dx = new LowPass();
    this.prev = null;
    this.lastT = null;
  }

  /** @param {number} value @param {number} t timestamp in seconds */
  filter(value, t) {
    const dt = this.lastT === null ? 1 / 30 : Math.max(1e-3, t - this.lastT);
    this.lastT = t;

    const rawSpeed = this.prev === null ? 0 : (value - this.prev) / dt;
    this.prev = value;
    const speed = this.dx.filter(rawSpeed, alphaFor(this.dCutoff, dt));
    const cutoff = this.minCutoff + this.beta * Math.abs(speed);
    return this.x.filter(value, alphaFor(cutoff, dt));
  }

  reset() {
    this.x.reset();
    this.dx.reset();
    this.prev = null;
    this.lastT = null;
  }
}

export class OneEuroPoint {
  constructor(opts) {
    this.fx = new OneEuroFilter(opts);
    this.fy = new OneEuroFilter(opts);
  }
  filter(x, y, t) {
    return { x: this.fx.filter(x, t), y: this.fy.filter(y, t) };
  }
  setParams({ minCutoff, beta }) {
    for (const f of [this.fx, this.fy]) {
      if (minCutoff !== undefined) f.minCutoff = minCutoff;
      if (beta !== undefined) f.beta = beta;
    }
  }
  reset() {
    this.fx.reset();
    this.fy.reset();
  }
}
