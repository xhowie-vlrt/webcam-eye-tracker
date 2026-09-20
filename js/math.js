// Small dense linear algebra + ridge regression.
// Everything here is plain arrays-of-arrays; the matrices involved are tiny
// (at most ~25x25), so readability beats micro-optimisation.

export function zeros(rows, cols) {
  return Array.from({ length: rows }, () => new Array(cols).fill(0));
}

/**
 * Solve A * X = B for X by Gauss-Jordan elimination with partial pivoting.
 * @param {number[][]} A square matrix (n x n), not mutated
 * @param {number[][]} B right-hand side (n x k), not mutated
 * @returns {number[][]} X (n x k)
 */
export function solve(A, B) {
  const n = A.length;
  const k = B[0].length;
  const M = A.map((row, i) => [...row, ...B[i]]);

  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(M[row][col]) > Math.abs(M[pivot][col])) pivot = row;
    }
    if (Math.abs(M[pivot][col]) < 1e-12) {
      throw new Error('singular matrix - not enough independent samples');
    }
    [M[col], M[pivot]] = [M[pivot], M[col]];

    const p = M[col][col];
    for (let j = col; j < n + k; j++) M[col][j] /= p;

    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const f = M[row][col];
      if (f === 0) continue;
      for (let j = col; j < n + k; j++) M[row][j] -= f * M[col][j];
    }
  }
  return M.map((row) => row.slice(n));
}

/**
 * Fit y = [1, z] * W with L2 penalty, where z is the z-scored feature vector.
 * The intercept is left unregularised.
 *
 * @param {number[][]} X samples (n x d), no intercept column
 * @param {number[][]} Y targets (n x k)
 * @param {number} lambda ridge strength, relative to one sample
 * @param {number[]} [weights] per-sample weights (used by the robust refit)
 * @returns {{mean:number[], std:number[], W:number[][]}}
 */
export function ridgeFit(X, Y, lambda = 1e-2, weights = null) {
  const n = X.length;
  if (n === 0) throw new Error('no samples');
  const d = X[0].length;
  const k = Y[0].length;
  const w = weights ?? null;
  let wSum = n;
  if (w) {
    wSum = 0;
    for (let i = 0; i < n; i++) wSum += w[i];
    if (!(wSum > 0)) throw new Error('all sample weights are zero');
  }

  const mean = new Array(d).fill(0);
  const std = new Array(d).fill(0);
  for (let i = 0; i < n; i++) {
    const wi = w ? w[i] : 1;
    for (let j = 0; j < d; j++) mean[j] += wi * X[i][j];
  }
  for (let j = 0; j < d; j++) mean[j] /= wSum;
  for (let i = 0; i < n; i++) {
    const wi = w ? w[i] : 1;
    for (let j = 0; j < d; j++) {
      const v = X[i][j] - mean[j];
      std[j] += wi * v * v;
    }
  }
  for (let j = 0; j < d; j++) {
    std[j] = Math.sqrt(std[j] / wSum);
    if (!(std[j] > 1e-9)) std[j] = 1; // constant column -> leave it alone
  }

  const D = d + 1;
  const A = zeros(D, D);
  const B = zeros(D, k);
  const z = new Array(D);

  for (let i = 0; i < n; i++) {
    const row = X[i];
    const wi = w ? w[i] : 1;
    if (wi === 0) continue;
    z[0] = 1;
    for (let j = 0; j < d; j++) z[j + 1] = (row[j] - mean[j]) / std[j];
    for (let a = 0; a < D; a++) {
      const za = wi * z[a];
      if (za === 0) continue;
      for (let b = a; b < D; b++) A[a][b] += za * z[b];
      for (let c = 0; c < k; c++) B[a][c] += za * Y[i][c];
    }
  }
  for (let a = 0; a < D; a++) for (let b = 0; b < a; b++) A[a][b] = A[b][a];
  // Scale the penalty by the effective sample count so lambda means the same
  // thing whether or not weights are in play.
  for (let a = 1; a < D; a++) A[a][a] += lambda * wSum;

  return { mean, std, W: solve(A, B) };
}

/** Predict with a model returned by {@link ridgeFit}. */
export function ridgePredict(model, x) {
  const { mean, std, W } = model;
  const d = mean.length;
  const k = W[0].length;
  const out = new Array(k);
  for (let c = 0; c < k; c++) out[c] = W[0][c];
  for (let j = 0; j < d; j++) {
    const z = (x[j] - mean[j]) / std[j];
    for (let c = 0; c < k; c++) out[c] += z * W[j + 1][c];
  }
  return out;
}
