// Accumulating gaze heatmap.
//
// Blobs are stamped into an offscreen greyscale buffer using alpha only; a
// colour ramp is applied in a single pass when rendering. That keeps repeated
// stamping cheap and avoids the muddy colours you get from blending gradients
// directly.

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
  constructor(canvas, { radius = 60, intensity = 0.09 } = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.radius = radius;
    this.intensity = intensity;
    this.buffer = document.createElement('canvas');
    this.bufferCtx = this.buffer.getContext('2d', { willReadFrequently: true });
    this.dirty = false;
    this.resize();
  }

  resize() {
    const w = Math.max(1, Math.floor(window.innerWidth));
    const h = Math.max(1, Math.floor(window.innerHeight));
    if (this.canvas.width === w && this.canvas.height === h) return;
    this.canvas.width = w;
    this.canvas.height = h;
    this.buffer.width = w;
    this.buffer.height = h;
    this.clear();
  }

  clear() {
    this.bufferCtx.clearRect(0, 0, this.buffer.width, this.buffer.height);
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.dirty = false;
  }

  add(x, y) {
    const r = this.radius;
    const g = this.bufferCtx.createRadialGradient(x, y, 0, x, y, r);
    g.addColorStop(0, `rgba(0,0,0,${this.intensity})`);
    g.addColorStop(1, 'rgba(0,0,0,0)');
    this.bufferCtx.fillStyle = g;
    this.bufferCtx.fillRect(x - r, y - r, r * 2, r * 2);
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
    this.ctx.putImageData(img, 0, 0);
  }
}
