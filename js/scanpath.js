// Scan path: the classic eye-tracking plot. Numbered circles sized by dwell
// time, joined in order, with a short fading trail of the raw signal so you can
// see how steady the tracking actually is.

const TRAIL_MAX = 90;
const FIXATION_MAX = 40;

export class ScanPath {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.trail = [];
    this.fixations = [];
    /** Result of the last accuracy check, drawn until something clears it. */
    this.validation = null;
    this.resize();
  }

  resize() {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = window.innerWidth;
    const h = window.innerHeight;
    this.canvas.width = Math.floor(w * dpr);
    this.canvas.height = Math.floor(h * dpr);
    this.canvas.style.width = `${w}px`;
    this.canvas.style.height = `${h}px`;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.draw();
  }

  clear() {
    this.trail = [];
    this.fixations = [];
    this.validation = null;
    this.draw();
  }

  /**
   * Show where each validation target was and where the model put it. Seeing
   * the error as a vector is far more actionable than a single mean: a
   * consistent direction means a posture shift, scatter means noise.
   * @param {Array<{x,y,predictedX,predictedY,error}>} points
   */
  showValidation(points) {
    this.validation = points;
    this.draw();
  }

  addGaze(x, y) {
    this.trail.push({ x, y });
    if (this.trail.length > TRAIL_MAX) this.trail.shift();
    this.draw();
  }

  addFixation(f) {
    this.fixations.push(f);
    if (this.fixations.length > FIXATION_MAX) this.fixations.shift();
    this.draw();
  }

  _drawValidation(ctx) {
    ctx.font = '600 11px ui-monospace, Menlo, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (const p of this.validation) {
      // Error vector: target -> what the model predicted.
      ctx.strokeStyle = 'rgba(248,113,113,0.9)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      ctx.lineTo(p.predictedX, p.predictedY);
      ctx.stroke();

      ctx.fillStyle = 'rgba(248,113,113,0.85)';
      ctx.beginPath();
      ctx.arc(p.predictedX, p.predictedY, 5, 0, Math.PI * 2);
      ctx.fill();

      // The true target.
      ctx.strokeStyle = 'rgba(90,200,250,0.95)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(p.x, p.y, 9, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(p.x - 13, p.y);
      ctx.lineTo(p.x + 13, p.y);
      ctx.moveTo(p.x, p.y - 13);
      ctx.lineTo(p.x, p.y + 13);
      ctx.stroke();

      ctx.fillStyle = 'rgba(230,235,243,0.9)';
      ctx.fillText(`${Math.round(p.error)}px`, p.x, p.y - 22);
    }
  }

  draw() {
    const ctx = this.ctx;
    const w = window.innerWidth;
    const h = window.innerHeight;
    ctx.clearRect(0, 0, w, h);

    // Raw trail, oldest fading out.
    ctx.lineWidth = 2;
    for (let i = 1; i < this.trail.length; i++) {
      const a = this.trail[i - 1];
      const b = this.trail[i];
      ctx.strokeStyle = `rgba(90,200,250,${(i / this.trail.length) * 0.35})`;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }

    if (this.validation) this._drawValidation(ctx);

    if (this.fixations.length === 0) return;

    // Saccade lines between fixation centroids.
    ctx.strokeStyle = 'rgba(230,235,243,0.35)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    this.fixations.forEach((f, i) => {
      if (i === 0) ctx.moveTo(f.x, f.y);
      else ctx.lineTo(f.x, f.y);
    });
    ctx.stroke();

    ctx.font = '600 12px ui-monospace, Menlo, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    this.fixations.forEach((f, i) => {
      // sqrt keeps a 2 s fixation from dwarfing a 200 ms one.
      const r = 10 + Math.sqrt(f.duration) * 0.8;
      const fresh = i / this.fixations.length;
      ctx.fillStyle = `rgba(250,220,80,${0.12 + fresh * 0.22})`;
      ctx.strokeStyle = `rgba(250,220,80,${0.35 + fresh * 0.45})`;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(f.x, f.y, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = `rgba(255,255,255,${0.5 + fresh * 0.5})`;
      ctx.fillText(String(i + 1), f.x, f.y);
    });
  }
}
