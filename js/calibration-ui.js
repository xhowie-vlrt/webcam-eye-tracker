// Self-contained calibration overlay.
//
// It builds its own DOM inside a shadow root, so an embedding page needs no
// markup and cannot accidentally restyle it. Two modes:
//
//   click   - classic discrete targets; the user clicks each one, which is the
//             cheapest possible proof that they are really looking at it.
//   pursuit - a dot that glides across the screen while samples stream in.
//             Far better screen coverage per second, at the cost of needing a
//             lag correction: the eye trails a moving target by ~100 ms, so a
//             sample taken at time t is paired with where the dot was at
//             t - lagMs, not where it is now.

const STYLE = `
:host { all: initial; }
.root {
  position: fixed; inset: 0; z-index: 2147483000;
  background: #05070b; color: #93a0b5;
  font: 14px/1.6 system-ui, -apple-system, "Hiragino Sans", "Noto Sans JP", sans-serif;
  cursor: default; overscroll-behavior: contain;
}
.hint { position: absolute; top: 8vh; left: 0; right: 0; text-align: center; }
.sub  { position: absolute; bottom: 6vh; left: 0; right: 0; text-align: center;
        font-size: 12px; opacity: .55; }
.bar  { position: absolute; bottom: 0; left: 0; height: 3px; background: #5ac8fa;
        width: 0; transition: width 120ms linear; }
.dot {
  position: absolute; top: 0; left: 0; width: 28px; height: 28px;
  margin: -14px 0 0 -14px; border-radius: 50%;
  background: #5ac8fa; box-shadow: 0 0 0 7px rgba(90,200,250,.16);
  cursor: pointer; will-change: transform;
}
.dot::after {
  content: ""; position: absolute; inset: 10px; border-radius: 50%; background: #04121c;
}
.dot.collecting {
  background: #4ade80; box-shadow: 0 0 0 11px rgba(74,222,128,.18); cursor: default;
}
.dot.smooth { transition: transform 240ms cubic-bezier(.4,0,.2,1); }
@media (prefers-reduced-motion: reduce) { .dot.smooth { transition: none; } }
`;

export class CalibrationOverlay {
  constructor(doc = document) {
    this.doc = doc;
    this.host = null;
    this.cancelled = false;
  }

  mount() {
    if (this.host) return;
    this.host = this.doc.createElement('div');
    this.host.setAttribute('data-eyetracker-overlay', '');
    const shadow = this.host.attachShadow({ mode: 'open' });
    shadow.innerHTML = `
      <style>${STYLE}</style>
      <div class="root" part="root">
        <p class="hint"></p>
        <div class="dot"></div>
        <p class="sub">Esc で中止</p>
        <div class="bar"></div>
      </div>`;
    this.el = {
      root: shadow.querySelector('.root'),
      hint: shadow.querySelector('.hint'),
      dot: shadow.querySelector('.dot'),
      sub: shadow.querySelector('.sub'),
      bar: shadow.querySelector('.bar'),
    };
    this.doc.body.appendChild(this.host);
    this._onKey = (e) => {
      if (e.key === 'Escape') this.cancel();
    };
    this.doc.addEventListener('keydown', this._onKey, true);
    this.cancelled = false;
  }

  unmount() {
    if (!this.host) return;
    this.doc.removeEventListener('keydown', this._onKey, true);
    this.host.remove();
    this.host = null;
  }

  cancel() {
    this.cancelled = true;
  }

  _setDot(x, y, smooth) {
    this.el.dot.classList.toggle('smooth', !!smooth);
    this.el.dot.style.transform = `translate(${x}px, ${y}px)`;
  }

  _progress(f) {
    this.el.bar.style.width = `${Math.round(f * 100)}%`;
  }

  /**
   * @param {Array<{x:number,y:number}>} points
   * @param {object} opts
   * @param {() => (number[]|null)} opts.getSample
   * @returns {Promise<Array<{x:number,y:number,vecs:number[][]}>>}
   */
  async runTargets(points, { getSample, label = 'キャリブレーション', collectMs = 900, settleMs = 250 } = {}) {
    this.mount();
    const out = [];
    try {
      for (let i = 0; i < points.length; i++) {
        if (this.cancelled) break;
        const p = points[i];
        this.el.hint.textContent = `${label} ${i + 1} / ${points.length} — 点を見つめたままクリック`;
        this.el.dot.classList.remove('collecting');
        this._setDot(p.x, p.y, i > 0);
        this._progress(i / points.length);

        await this._waitForClick();
        if (this.cancelled) break;

        this.el.dot.classList.add('collecting');
        await this._sleep(settleMs); // let the saccade land before recording
        const vecs = await this._collectFor(collectMs, getSample);
        if (vecs.length) out.push({ x: p.x, y: p.y, vecs });
      }
      this._progress(1);
    } finally {
      this.unmount();
    }
    return this.cancelled ? [] : out;
  }

  /**
   * Smooth-pursuit pass. Returns one group per path segment.
   * @returns {Promise<Array<{x:number,y:number,vecs:number[][],samples:Array}>>}
   */
  async runPursuit({
    waypoints,
    getSample,
    speed = 420, // px/s: slow enough that pursuit stays smooth
    lagMs = 120,
    label = 'スムースパスート キャリブレーション',
    settleMs = 700,
  }) {
    this.mount();
    this.el.hint.textContent = `${label} — 動く点を目で追ってください`;
    this.el.dot.classList.add('collecting');
    this._setDot(waypoints[0].x, waypoints[0].y, false);

    const total = pathLength(waypoints);
    /** Ring of recent target positions, so we can look up where it *was*. */
    const history = [];
    const raw = [];

    try {
      await this._sleep(settleMs);
      if (this.cancelled) return [];

      const started = performance.now();
      await new Promise((resolve) => {
        const step = () => {
          if (this.cancelled) return resolve();
          const now = performance.now();
          const travelled = ((now - started) / 1000) * speed;
          if (travelled >= total) return resolve();

          const { x, y, segment } = pointAt(waypoints, travelled);
          this._setDot(x, y, false);
          this._progress(travelled / total);
          history.push({ t: now, x, y, segment });
          if (history.length > 600) history.shift();

          const vec = getSample();
          if (vec) {
            const target = lookup(history, now - lagMs);
            if (target) raw.push({ vec, x: target.x, y: target.y, segment: target.segment });
          }
          requestAnimationFrame(step);
        };
        requestAnimationFrame(step);
      });
      this._progress(1);
    } finally {
      this.unmount();
    }

    if (this.cancelled) return [];

    // Group by path segment so cross-validation holds out whole stretches of
    // the path rather than interleaved, near-identical frames.
    const groups = new Map();
    for (const s of raw) {
      if (!groups.has(s.segment)) groups.set(s.segment, []);
      groups.get(s.segment).push(s);
    }
    return [...groups.values()]
      .filter((g) => g.length >= 5)
      .map((g) => ({
        x: g.reduce((a, s) => a + s.x, 0) / g.length,
        y: g.reduce((a, s) => a + s.y, 0) / g.length,
        vecs: g.map((s) => s.vec),
        samples: g,
      }));
  }

  _waitForClick() {
    return new Promise((resolve) => {
      const done = () => {
        this.el.dot.removeEventListener('pointerdown', done);
        clearInterval(poll);
        resolve();
      };
      this.el.dot.addEventListener('pointerdown', done, { once: true });
      const poll = setInterval(() => {
        if (this.cancelled) done();
      }, 50);
    });
  }

  _collectFor(ms, getSample) {
    return new Promise((resolve) => {
      const vecs = [];
      const until = performance.now() + ms;
      const step = () => {
        if (this.cancelled) return resolve(vecs);
        const vec = getSample();
        if (vec) vecs.push(vec);
        const remaining = until - performance.now();
        if (remaining > 0) requestAnimationFrame(step);
        else resolve(vecs);
      };
      requestAnimationFrame(step);
    });
  }

  _sleep(ms) {
    return new Promise((r) => setTimeout(r, ms));
  }
}

// -- path helpers (exported for testing) -------------------------------------

export function pathLength(waypoints) {
  let total = 0;
  for (let i = 1; i < waypoints.length; i++) {
    total += Math.hypot(
      waypoints[i].x - waypoints[i - 1].x,
      waypoints[i].y - waypoints[i - 1].y
    );
  }
  return total;
}

/** Position `distance` along the polyline, plus which segment it fell in. */
export function pointAt(waypoints, distance) {
  let remaining = distance;
  for (let i = 1; i < waypoints.length; i++) {
    const a = waypoints[i - 1];
    const b = waypoints[i];
    const len = Math.hypot(b.x - a.x, b.y - a.y);
    if (remaining <= len || i === waypoints.length - 1) {
      const f = len === 0 ? 0 : Math.min(1, remaining / len);
      return { x: a.x + (b.x - a.x) * f, y: a.y + (b.y - a.y) * f, segment: i - 1 };
    }
    remaining -= len;
  }
  const last = waypoints[waypoints.length - 1];
  return { x: last.x, y: last.y, segment: waypoints.length - 2 };
}

function lookup(history, t) {
  // History is monotonic in time; walk back from the end.
  for (let i = history.length - 1; i >= 0; i--) {
    if (history[i].t <= t) return history[i];
  }
  return history[0] ?? null;
}

/** Serpentine path covering the viewport: 4 rows, alternating direction. */
export function serpentine(width, height, { rows = 4, inset = 0.07 } = {}) {
  const pts = [];
  const x0 = width * inset;
  const x1 = width * (1 - inset);
  for (let r = 0; r < rows; r++) {
    const y = height * (inset + (r / (rows - 1)) * (1 - 2 * inset));
    const leftToRight = r % 2 === 0;
    pts.push({ x: leftToRight ? x0 : x1, y });
    pts.push({ x: leftToRight ? x1 : x0, y });
  }
  return pts;
}
