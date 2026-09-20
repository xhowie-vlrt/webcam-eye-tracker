// Target layouts for calibration and validation.

/** Evenly spaced grid, inset from the edges so the dot is never clipped. */
export function gridPoints(cols = 3, rows = 3, { inset = 0.07, width, height } = {}) {
  const w = width ?? window.innerWidth;
  const h = height ?? window.innerHeight;
  const span = 1 - 2 * inset;
  const at = (i, n) => (n === 1 ? 0.5 : inset + (i / (n - 1)) * span);
  const points = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      points.push({ x: at(c, cols) * w, y: at(r, rows) * h });
    }
  }
  return points;
}

/**
 * Validation targets deliberately sit *between* the calibration grid nodes,
 * so the reported error measures interpolation rather than memorisation.
 */
export function validationPoints({ width, height } = {}) {
  const w = width ?? window.innerWidth;
  const h = height ?? window.innerHeight;
  return [
    [0.2, 0.2],
    [0.8, 0.22],
    [0.5, 0.5],
    [0.22, 0.8],
    [0.78, 0.78],
    [0.5, 0.15],
    [0.15, 0.5],
    [0.85, 0.5],
    [0.5, 0.85],
  ].map(([fx, fy]) => ({ x: fx * w, y: fy * h }));
}

export function shuffle(items, rand = Math.random) {
  const out = [...items];
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}
