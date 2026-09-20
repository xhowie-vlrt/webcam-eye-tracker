// Velocity-threshold fixation detection (I-VT).
//
// Raw gaze samples are the wrong unit for most applications: what you usually
// want is "the user looked here, for this long". I-VT splits the stream into
// fixations (slow) and saccades (fast), which also hides a lot of per-frame
// noise that no amount of smoothing would remove.

export class FixationDetector {
  /**
   * @param {object} [opts]
   * @param {number} [opts.velocityThreshold] px/s above which we call it a saccade
   * @param {number} [opts.minDurationMs] shorter candidates are discarded
   * @param {number} [opts.maxGapMs] a longer gap ends the fixation. Spontaneous
   *   blinks run 100-400 ms and no samples are emitted during one, so the
   *   default bridges a blink rather than splitting one fixation into two.
   */
  constructor({
    velocityThreshold = 600,
    minDurationMs = 100,
    maxGapMs = 450,
  } = {}) {
    this.velocityThreshold = velocityThreshold;
    this.minDurationMs = minDurationMs;
    this.maxGapMs = maxGapMs;
    this.reset();
  }

  reset() {
    this.buffer = [];
    this.previous = null;
    /** The fixation currently being accumulated, exposed for live display. */
    this.current = null;
  }

  /**
   * Feed one gaze sample.
   * @param {number} x
   * @param {number} y
   * @param {number} t timestamp in ms
   * @returns {null|{x:number,y:number,start:number,end:number,duration:number,samples:number,dispersion:number}}
   *   the fixation that just *ended*, if any
   */
  add(x, y, t) {
    const prev = this.previous;
    this.previous = { x, y, t };

    if (!prev) {
      this.buffer = [{ x, y, t }];
      return null;
    }

    const dt = t - prev.t;
    if (dt <= 0) return null;

    if (dt > this.maxGapMs) {
      const ended = this._close();
      this.buffer = [{ x, y, t }];
      return ended;
    }

    const velocity = (Math.hypot(x - prev.x, y - prev.y) * 1000) / dt;
    if (velocity > this.velocityThreshold) {
      const ended = this._close();
      this.buffer = [{ x, y, t }];
      return ended;
    }

    this.buffer.push({ x, y, t });
    this.current = this._summarise(this.buffer);
    return null;
  }

  /** Force the in-progress fixation to end, e.g. when tracking stops. */
  flush() {
    return this._close();
  }

  _close() {
    const f = this._summarise(this.buffer);
    this.buffer = [];
    this.current = null;
    return f && f.duration >= this.minDurationMs ? f : null;
  }

  _summarise(points) {
    if (points.length < 2) return null;
    let sx = 0;
    let sy = 0;
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (const p of points) {
      sx += p.x;
      sy += p.y;
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }
    const start = points[0].t;
    const end = points[points.length - 1].t;
    return {
      x: sx / points.length,
      y: sy / points.length,
      start,
      end,
      duration: end - start,
      samples: points.length,
      // Bounding-box dispersion: a cheap confidence proxy.
      dispersion: maxX - minX + (maxY - minY),
    };
  }
}
