// Public API. Everything an embedding page needs is here:
//
//   const et = new EyeTracker();
//   await et.start();
//   await et.calibrate();
//   et.on('gaze', ({ x, y }) => ...);
//
// The class owns the camera, the landmarker, the model and the filters, and
// emits plain objects. It creates its own <video> unless one is supplied, so
// a host page can integrate it with no markup at all.

import { FaceTracker } from './tracker.js';
import { buildFeatures } from './features.js';
import { GazeModel, pixelsToDegrees } from './gaze.js';
import { OneEuroPoint } from './filter.js';
import { FixationDetector } from './fixation.js';
import { CalibrationOverlay, serpentine } from './calibration-ui.js';
import { gridPoints, validationPoints, shuffle } from './points.js';

// Bootstrap only, for the ~1 s before the rolling median has enough frames.
// Measured against MediaPipe's own test portrait (test/e2e/landmarks.mjs), a
// real open eye sits at EAR ~0.22, so the commonly quoted 0.17 leaves almost
// no margin. Erring low means a blink in that first second may be missed,
// which costs far less than rejecting every frame as a blink.
const BLINK_EAR = 0.13;
const BLINK_EAR_RATIO = 0.6; // of the median open-eye ratio
const BLINK_EAR_RANGE = [0.1, 0.25];
const EAR_WINDOW = 300; // ~10 s at 30 fps
const EAR_RECOMPUTE_EVERY = 30;
const DRIFT_WEIGHT = 0.25; // a click is a weaker signal than a calibration point
const DRIFT_REFIT_EVERY = 4;

export class EyeTracker {
  /**
   * @param {object} [opts]
   * @param {HTMLVideoElement} [opts.video] reuse an existing element
   * @param {number} [opts.smoothing] 0 (responsive) .. 1 (very smooth)
   * @param {boolean} [opts.driftCorrection] learn from clicks while running
   * @param {object} [opts.fixation] options for {@link FixationDetector}
   */
  constructor({
    video = null,
    smoothing = 0.55,
    driftCorrection = false,
    fixation = {},
  } = {}) {
    this.video = video ?? createHiddenVideo();
    this._ownsVideo = !video;
    this.tracker = new FaceTracker(this.video);
    this.model = new GazeModel();
    this.smoother = new OneEuroPoint();
    this.fixations = new FixationDetector(fixation);
    this.overlay = new CalibrationOverlay();

    this.listeners = new Map();
    this.latest = null; // { vec, ear, landmarks, timestamp }
    this.lastGaze = null;
    this.blinking = false;
    this.calibrating = false;
    this._driftPending = 0;
    this._driftGroup = 0;
    this._calEars = [];
    this._earWindow = [];
    this._earTick = 0;
    this._liveBlinkThreshold = null;
    this._poseOk = true;

    this.setSmoothing(smoothing);
    this.tracker.onResult = (r) => this._onFrame(r);

    this._onPointerDown = (e) => this._maybeDriftCorrect(e);
    this.driftCorrection = false;
    if (driftCorrection) this.setDriftCorrection(true);
  }

  // -- lifecycle ------------------------------------------------------------

  async start() {
    this._emit('status', { phase: 'loading', message: 'loading model' });
    try {
      await this.tracker.start();
      this._emit('status', {
        phase: 'running',
        message: this.calibrated ? 'tracking' : 'needs calibration',
      });
    } catch (err) {
      this._emit('error', err);
      throw err;
    }
  }

  stop() {
    this.tracker.stop();
    const last = this.fixations.flush();
    if (last) this._emit('fixation', last);
    this.latest = null;
    this._emit('status', { phase: 'stopped', message: 'camera stopped' });
  }

  /** Release the camera, listeners and (if we made it) the video element. */
  destroy() {
    this.stop();
    this.setDriftCorrection(false);
    this.overlay.unmount();
    this.listeners.clear();
    if (this._ownsVideo) this.video.remove();
  }

  get running() {
    return this.tracker.running;
  }

  get calibrated() {
    return this.model.ready;
  }

  /**
   * Eye-aspect ratio below which we call it a blink.
   *
   * A fixed constant misfires badly on narrow eyes or a camera mounted well
   * above or below eye level: either every frame reads as a blink - which
   * would make the very first calibration collect nothing and fail - or none
   * do. So the threshold tracks a rolling median from the first seconds of
   * video, and the calibrated value takes over once it exists.
   */
  get blinkThreshold() {
    return this.model.blinkThreshold ?? this._liveBlinkThreshold ?? BLINK_EAR;
  }

  get fps() {
    return this.tracker.fps;
  }

  // -- events ---------------------------------------------------------------

  /**
   * @param {'gaze'|'fixation'|'face'|'quality'|'status'|'error'} event
   * @returns {() => void} unsubscribe
   */
  on(event, handler) {
    if (!this.listeners.has(event)) this.listeners.set(event, new Set());
    this.listeners.get(event).add(handler);
    return () => this.listeners.get(event)?.delete(handler);
  }

  _emit(event, payload) {
    for (const fn of this.listeners.get(event) ?? []) {
      try {
        fn(payload);
      } catch (err) {
        console.error(`eyetracker: "${event}" handler threw`, err);
      }
    }
  }

  // -- per-frame ------------------------------------------------------------

  _onFrame({ landmarks, timestamp }) {
    if (!landmarks) {
      this.latest = null;
      this._emit('face', { found: false, fps: this.tracker.fps });
      return;
    }

    const aspect = (this.video.videoWidth || 16) / (this.video.videoHeight || 9);
    const f = buildFeatures(landmarks, aspect);
    this._trackEar(f.ear);
    this.blinking = f.ear < this.blinkThreshold;
    this.latest = { vec: f.vec, ear: f.ear, landmarks, timestamp };
    if (this.calibrating) this._calEars.push(f.ear);
    this._emit('face', {
      found: true,
      landmarks,
      ear: f.ear,
      blink: this.blinking,
      head: f.head,
      fps: this.tracker.fps,
    });

    if (!this.model.ready) return;
    if (this.blinking) {
      // Hold the last position rather than reporting where a half-closed lid
      // appears to point.
      if (this.lastGaze) {
        this._emit('gaze', { ...this.lastGaze, blink: true, timestamp });
      }
      return;
    }

    // Flag when the head has wandered outside the range the model was fitted
    // on: past that point the prediction is extrapolation, and the honest
    // thing is to say so rather than keep drawing a confident dot.
    const quality = this.model.poseQuality(f.vec);
    if (quality.ok !== this._poseOk) {
      this._poseOk = quality.ok;
      this._emit('quality', quality);
    }

    const raw = this.model.predict(f.vec);
    const s = this.smoother.filter(raw.x, raw.y, timestamp / 1000);
    const x = clamp(s.x, 0, window.innerWidth);
    const y = clamp(s.y, 0, window.innerHeight);

    const ended = this.fixations.add(x, y, timestamp);
    if (ended) this._emit('fixation', ended);

    this.lastGaze = {
      x,
      y,
      rawX: raw.x,
      rawY: raw.y,
      blink: false,
      quality,
      fixation: this.fixations.current,
      timestamp,
    };
    this._emit('gaze', this.lastGaze);
  }

  _trackEar(ear) {
    this._earWindow.push(ear);
    if (this._earWindow.length > EAR_WINDOW) this._earWindow.shift();
    // Sorting every frame would be wasteful; the estimate moves slowly.
    if (++this._earTick % EAR_RECOMPUTE_EVERY === 0) {
      this._liveBlinkThreshold = learnBlinkThreshold(this._earWindow);
    }
  }

  /** Latest feature vector, or null while blinking / no face. */
  sample() {
    return this.latest && !this.blinking ? this.latest.vec : null;
  }

  // -- calibration ----------------------------------------------------------

  /**
   * @param {object} [opts]
   * @param {'click'|'pursuit'|'both'} [opts.mode]
   * @param {number} [opts.cols] grid width for click mode
   * @param {number} [opts.rows] grid height for click mode
   * @returns {Promise<null|{train:object, cv:number, lambda:number, groups:number, samples:number}>}
   *   null if the user cancelled
   */
  async calibrate({ mode = 'click', cols = 3, rows = 3, pursuitRows = 4, speed = 420 } = {}) {
    if (!this.running) throw new Error('start() the tracker before calibrating');
    this.calibrating = true;
    this._calEars = [];
    this._emit('status', { phase: 'calibrating', message: mode });
    try {
      const groups = [];
      if (mode === 'click' || mode === 'both') {
        const pts = shuffle(gridPoints(cols, rows));
        const got = await this.overlay.runTargets(pts, {
          getSample: () => this.sample(),
        });
        if (got.length < pts.length) return null;
        groups.push(...got);
      }
      if (mode === 'pursuit' || mode === 'both') {
        const got = await this.overlay.runPursuit({
          waypoints: serpentine(window.innerWidth, window.innerHeight, { rows: pursuitRows }),
          getSample: () => this.sample(),
          speed,
        });
        if (got.length === 0) return null;
        groups.push(...got);
      }

      this.model.clear();
      groups.forEach((g, i) => {
        if (g.samples) {
          // Pursuit groups carry a per-sample target; use it, not the centroid.
          for (const s of g.samples) this.model.addSample(s.vec, s.x, s.y, { group: i });
        } else {
          for (const vec of g.vecs) this.model.addSample(vec, g.x, g.y, { group: i });
        }
      });

      const report = this.model.fit();
      // The whole calibration is a longer, more representative sample than
      // the rolling window, so prefer it once we have it.
      this.model.blinkThreshold =
        learnBlinkThreshold(this._calEars) ?? this._liveBlinkThreshold;
      this.model.save();
      this.smoother.reset();
      this.fixations.reset();
      this._emit('status', { phase: 'running', message: 'calibrated' });
      return { ...report, groups: groups.length, samples: this.model.sampleCount };
    } finally {
      this.calibrating = false;
    }
  }

  /**
   * Measure accuracy on targets the model was never trained on.
   * @returns {Promise<null|{mean:number, p95:number, max:number, degrees:number,
   *   points:Array<{x,y,predictedX,predictedY,error}>}>}
   */
  async validate({ points = null } = {}) {
    if (!this.model.ready) throw new Error('calibrate before validating');
    this.calibrating = true;
    this._emit('status', { phase: 'validating', message: '' });
    try {
      const pts = shuffle(points ?? validationPoints());
      const got = await this.overlay.runTargets(pts, {
        getSample: () => this.sample(),
        label: '精度チェック',
        collectMs: 800,
      });
      if (got.length < pts.length) return null;

      const errors = got.map((g) => {
        let sx = 0;
        let sy = 0;
        for (const vec of g.vecs) {
          const p = this.model.predict(vec);
          sx += p.x;
          sy += p.y;
        }
        const px = sx / g.vecs.length;
        const py = sy / g.vecs.length;
        return {
          x: g.x,
          y: g.y,
          predictedX: px,
          predictedY: py,
          error: Math.hypot(px - g.x, py - g.y),
        };
      });
      const sorted = errors.map((e) => e.error).sort((a, b) => a - b);
      const mean = sorted.reduce((a, b) => a + b, 0) / sorted.length;
      const result = {
        mean,
        p95: sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95))],
        max: sorted[sorted.length - 1],
        degrees: pixelsToDegrees(mean),
        points: errors,
      };
      this._emit('status', { phase: 'running', message: 'validated' });
      return result;
    } finally {
      this.calibrating = false;
    }
  }

  cancelCalibration() {
    this.overlay.cancel();
  }

  // -- tuning ---------------------------------------------------------------

  /** @param {number} v 0 = responsive and jumpy, 1 = smooth and laggy */
  setSmoothing(v) {
    const t = clamp(v, 0, 1);
    this.smoothing = t;
    this.smoother.setParams({ minCutoff: 8 * Math.pow(0.02 / 8, t), beta: 0.02 });
  }

  /**
   * Drift correction: when the user clicks, assume they were looking at the
   * cursor and fold that in as a low-weight sample. This is what keeps a
   * session usable for longer than a few minutes without recalibrating.
   */
  setDriftCorrection(on) {
    if (on === this.driftCorrection) return;
    this.driftCorrection = on;
    if (on) window.addEventListener('pointerdown', this._onPointerDown, true);
    else window.removeEventListener('pointerdown', this._onPointerDown, true);
  }

  _maybeDriftCorrect(e) {
    if (!this.model.ready || this.calibrating || this.blinking) return;
    const vec = this.sample();
    if (!vec) return;
    this.model.addSample(vec, e.clientX, e.clientY, {
      group: `drift:${this._driftGroup++}`,
      weight: DRIFT_WEIGHT,
    });
    if (++this._driftPending >= DRIFT_REFIT_EVERY) {
      this._driftPending = 0;
      this.model.refit();
      this._emit('status', { phase: 'running', message: 'drift corrected' });
    }
  }

  // -- persistence ----------------------------------------------------------

  loadModel() {
    return this.model.load();
  }

  clearModel() {
    this.model.clear();
    GazeModel.forget();
    this.smoother.reset();
    this.fixations.reset();
  }
}

/**
 * Blinks are brief, so the median eye-aspect ratio over a whole calibration
 * is a good estimate of this user's open-eye value at this camera angle.
 */
function learnBlinkThreshold(ears) {
  if (ears.length < 30) return null;
  const sorted = [...ears].sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)];
  return clamp(median * BLINK_EAR_RATIO, BLINK_EAR_RANGE[0], BLINK_EAR_RANGE[1]);
}

function createHiddenVideo() {
  const v = document.createElement('video');
  v.playsInline = true;
  v.muted = true;
  v.style.cssText =
    'position:fixed;width:1px;height:1px;opacity:0;pointer-events:none;top:-10px;left:-10px';
  document.body.appendChild(v);
  return v;
}

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

export { pixelsToDegrees, gridPoints, validationPoints, serpentine, learnBlinkThreshold };
