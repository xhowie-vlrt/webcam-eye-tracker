// Calibration + validation flow: show a target, wait for the user to click it
// (clicking is the cheapest proof that they are actually looking at it), then
// record feature vectors for a moment before moving on.

const COLLECT_MS = 900;
const SETTLE_MS = 250;

/** 3x3 grid, inset from the edges so the dot is never clipped. */
export function gridPoints(cols = 3, rows = 3, inset = 0.08) {
  const points = [];
  const span = 1 - 2 * inset;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      points.push({
        x: (inset + (cols === 1 ? 0.5 : (c / (cols - 1)) * span)) * window.innerWidth,
        y: (inset + (rows === 1 ? 0.5 : (r / (rows - 1)) * span)) * window.innerHeight,
      });
    }
  }
  return points;
}

/** Five points that deliberately sit *between* the calibration grid nodes. */
export function validationPoints() {
  const w = window.innerWidth;
  const h = window.innerHeight;
  return [
    { x: 0.25 * w, y: 0.25 * h },
    { x: 0.75 * w, y: 0.25 * h },
    { x: 0.5 * w, y: 0.5 * h },
    { x: 0.25 * w, y: 0.75 * h },
    { x: 0.75 * w, y: 0.75 * h },
  ];
}

export function shuffle(items) {
  const out = [...items];
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

export class CalibrationRunner {
  /**
   * @param {object} deps
   * @param {HTMLElement} deps.overlay full-screen container, hidden by default
   * @param {HTMLElement} deps.dot     the target the user clicks
   * @param {HTMLElement} deps.hint    progress text
   * @param {() => (number[]|null)} deps.getSample latest feature vector, or null
   */
  constructor({ overlay, dot, hint, getSample }) {
    this.overlay = overlay;
    this.dot = dot;
    this.hint = hint;
    this.getSample = getSample;
    this.cancelled = false;
  }

  cancel() {
    this.cancelled = true;
  }

  /**
   * @returns {Promise<Array<{x:number, y:number, vecs:number[][]}>>}
   */
  async run(points, label = 'キャリブレーション') {
    this.cancelled = false;
    this.overlay.classList.remove('hidden');
    document.body.classList.add('calibrating');

    const onKey = (e) => {
      if (e.key === 'Escape') this.cancel();
    };
    window.addEventListener('keydown', onKey);

    const collected = [];
    try {
      for (let i = 0; i < points.length; i++) {
        if (this.cancelled) break;
        const p = points[i];
        this.hint.textContent = `${label} ${i + 1} / ${points.length} — 点を見つめたままクリック`;
        this.dot.style.left = `${p.x}px`;
        this.dot.style.top = `${p.y}px`;
        this.dot.classList.remove('collecting');

        await this._waitForClick();
        if (this.cancelled) break;

        this.dot.classList.add('collecting');
        await wait(SETTLE_MS); // let the saccade land before recording
        const vecs = await this._collect(COLLECT_MS);
        if (vecs.length > 0) collected.push({ x: p.x, y: p.y, vecs });
      }
    } finally {
      window.removeEventListener('keydown', onKey);
      this.overlay.classList.add('hidden');
      this.dot.classList.remove('collecting');
      document.body.classList.remove('calibrating');
    }

    return this.cancelled ? [] : collected;
  }

  _waitForClick() {
    return new Promise((resolve) => {
      const done = () => {
        this.dot.removeEventListener('pointerdown', done);
        clearInterval(poll);
        resolve();
      };
      this.dot.addEventListener('pointerdown', done, { once: true });
      // Escape resolves too, so the loop above can see `cancelled`.
      const poll = setInterval(() => {
        if (this.cancelled) done();
      }, 50);
    });
  }

  _collect(durationMs) {
    return new Promise((resolve) => {
      const vecs = [];
      const until = performance.now() + durationMs;
      const step = () => {
        if (this.cancelled) return resolve(vecs);
        const vec = this.getSample();
        if (vec) vecs.push(vec);
        if (performance.now() < until) requestAnimationFrame(step);
        else resolve(vecs);
      };
      requestAnimationFrame(step);
    });
  }
}

function wait(ms) {
  return new Promise((r) => setTimeout(r, ms));
}
