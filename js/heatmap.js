// Accumulating gaze heatmap.
//
// Blobs are stamped into an offscreen greyscale buffer using alpha only; a
// colour ramp is applied in one pass when rendering. That keeps repeated
// stamping cheap and avoids the muddy colours you get from blending gradients
// directly.
//
// The buffer is capped well below screen resolution. A heatmap is a blurred
// density estimate, so the detail is not missed - but re-colourising a 4K
// viewport would mean walking 33 MB of pixels several times a second.

const MAX_BUFFER_EDGE = 960;

const RAMP = [
  [0.0, [0, 0, 0, 0]],
  [0.2, [60, 120, 255, 90]],
  [0.45, [40, 220, 200, 150]],
  [0.7, [250, 220, 80, 200]],
  [1.0, [255, 70, 60, 230]],
];

function buildLut() {
  const lut = new Uint8ClampedArray(256 * 4);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let a = RAMP[0];
    let b = RAMP[RAMP.length - 1];
    for (let k = 0; k < RAMP.length - 1; k++) {
      if (t >= RAMP[k][0] && t <= RAMP[k + 1][0]) {
        a = RAMP[k];
        b = RAMP[k + 1];
        break;
      }
    }
    const span = b[0] - a[0] || 1;
    const f = (t - a[0]) / span;
    for (let c = 0; c < 4; c++) {
      lut[i * 4 + c] = a[1][c] + (b[1][c] - a[1][c]) * f;
    }
  }
  return lut;
}

const LUT = buildLut();

export class Heatmap {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {object} [opts]
   * @param {number} [opts.radius] blob radius in CSS pixels
   * @param {number} [opts.intensity] alpha added per sample
   */
  constructor(canvas, { radius = 60, intensity = 0.09 } = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.radius = radius;
    this.intensity = intensity;
    this.scale = 1;

    this.buffer = document.createElement('canvas');
    this.bufferCtx = this.buffer.getContext('2d', { willReadFrequently: true });
    this.tint = document.createElement('canvas');
    this.tintCtx = this.tint.getContext('2d');

    this.dirty = false;
    this.width = 0;
    this.height = 0;
    this.resize();
  }

  resize() {
    const w = Math.max(1, Math.floor(window.innerWidth));
    const h = Math.max(1, Math.floor(window.innerHeight));
    // Compare against our own last size, not the canvas element's: the canvas
    // may already carry those dimensions (a default 300x150, or another
    // instance's) while our buffers are still unsized.
    if (this.width === w && this.height === h) return;
    this.width = w;
    this.height = h;

    this.canvas.width = w;
    this.canvas.height = h;
    this.scale = Math.min(1, MAX_BUFFER_EDGE / Math.max(w, h));

    const bw = Math.max(1, Math.round(w * this.scale));
    const bh = Math.max(1, Math.round(h * this.scale));
    this.buffer.width = bw;
    this.buffer.height = bh;
    this.tint.width = bw;
    this.tint.height = bh;

    // Scaling up a small buffer is what gives the heatmap its softness.
    this.ctx.imageSmoothingEnabled = true;
    this.ctx.imageSmoothingQuality = 'high';
    this.clear();
  }

  clear() {
    this.bufferCtx.clearRect(0, 0, this.buffer.width, this.buffer.height);
    this.tintCtx.clearRect(0, 0, this.tint.width, this.tint.height);
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.dirty = false;
  }

  /** @param {number} x @param {number} y CSS pixel coordinates */
  add(x, y) {
    const bx = x * this.scale;
    const by = y * this.scale;
    const r = Math.max(1, this.radius * this.scale);
    const g = this.bufferCtx.createRadialGradient(bx, by, 0, bx, by, r);
    g.addColorStop(0, `rgba(0,0,0,${this.intensity})`);
    g.addColorStop(1, 'rgba(0,0,0,0)');
    this.bufferCtx.fillStyle = g;
    this.bufferCtx.fillRect(bx - r, by - r, r * 2, r * 2);
    this.dirty = true;
  }

  render() {
    if (!this.dirty) return;
    this.dirty = false;
    const { width: w, height: h } = this.buffer;
    const img = this.bufferCtx.getImageData(0, 0, w, h);
    const data = img.data;
    for (let i = 0; i < data.length; i += 4) {
      const a = data[i + 3];
      if (a === 0) continue;
      const o = a * 4;
      data[i] = LUT[o];
      data[i + 1] = LUT[o + 1];
      data[i + 2] = LUT[o + 2];
      data[i + 3] = LUT[o + 3];
    }
    this.tintCtx.putImageData(img, 0, 0);
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.ctx.drawImage(this.tint, 0, 0, this.canvas.width, this.canvas.height);
  }
}
