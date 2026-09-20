// Turn a MediaPipe FaceLandmarker result into the feature vector the gaze
// regressor is trained on.
//
// The core signal is the iris offset inside each eye socket. That alone only
// works if the head never moves, so we add cheap landmark-derived head-pose
// proxies (yaw / pitch / roll / scale / position) plus a few cross terms, and
// let ridge regression work out how much each one matters for this user.

/** Landmark indices in the 478-point FaceLandmarker mesh (iris included). */
export const IDX = {
  // "A"/"B" rather than left/right: which one is which depends on mirroring,
  // and the regressor does not care as long as we stay consistent.
  eyeA: { outer: 33, inner: 133, upper: 159, lower: 145, iris: 468 },
  eyeB: { outer: 263, inner: 362, upper: 386, lower: 374, iris: 473 },
  nose: 1,
  chin: 152,
  brow: 10,
};

/** All indices we actually draw in the debug overlay. */
export const OVERLAY_POINTS = [
  ...Object.values(IDX.eyeA),
  ...Object.values(IDX.eyeB),
  IDX.nose,
  IDX.chin,
  IDX.brow,
];

// Landmarks are normalised to [0,1] against width and height separately, so x
// has to be scaled by the aspect ratio before any distance is meaningful.
const px = (p, aspect) => ({ x: p.x * aspect, y: p.y });
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

function eyeMetrics(lm, e, aspect) {
  const outer = px(lm[e.outer], aspect);
  const inner = px(lm[e.inner], aspect);
  const upper = px(lm[e.upper], aspect);
  const lower = px(lm[e.lower], aspect);
  const iris = px(lm[e.iris], aspect);

  const width = dist(outer, inner) || 1e-6;
  const cx = (outer.x + inner.x) / 2;
  const cy = (upper.y + lower.y) / 2;

  return {
    // Iris offset from the socket centre, in eye-widths.
    nx: (iris.x - cx) / width,
    ny: (iris.y - cy) / width,
    // Eye aspect ratio: small means the lid is closed.
    ear: dist(upper, lower) / width,
  };
}

function headMetrics(lm, aspect) {
  const a = px(lm[IDX.eyeA.outer], aspect);
  const b = px(lm[IDX.eyeB.outer], aspect);
  const nose = px(lm[IDX.nose], aspect);
  const chin = px(lm[IDX.chin], aspect);
  const brow = px(lm[IDX.brow], aspect);

  const scale = dist(a, b) || 1e-6; // inter-ocular distance ~ 1 / camera distance
  const midX = (a.x + b.x) / 2;
  const midY = (a.y + b.y) / 2;

  return {
    // Turning the head makes the nose closer to one eye corner than the other.
    yaw: (dist(nose, a) - dist(nose, b)) / scale,
    // Nodding moves the nose relative to the eye line, and shortens the face.
    pitch: (nose.y - midY) / scale,
    tilt: dist(brow, chin) / scale,
    roll: Math.atan2(b.y - a.y, b.x - a.x),
    scale,
    headX: midX,
    headY: midY,
  };
}

/**
 * @param {Array<{x:number,y:number,z:number}>} landmarks 478 normalised points
 * @param {number} aspect video width / height
 * @returns {{vec:number[], ear:number, head:object, eyes:object}}
 */
export function buildFeatures(landmarks, aspect) {
  const A = eyeMetrics(landmarks, IDX.eyeA, aspect);
  const B = eyeMetrics(landmarks, IDX.eyeB, aspect);
  const h = headMetrics(landmarks, aspect);

  // Mean iris offset: the single most informative pair of numbers.
  const mx = (A.nx + B.nx) / 2;
  const my = (A.ny + B.ny) / 2;

  const vec = [
    A.nx, A.ny, B.nx, B.ny,
    mx, my,
    mx * mx, my * my, mx * my,
    h.yaw, h.pitch, h.roll, h.scale, h.tilt, h.headX, h.headY,
    h.yaw * h.yaw, h.pitch * h.pitch,
    mx * h.yaw, my * h.pitch,
    mx * h.scale, my * h.scale,
    mx * h.headX, my * h.headY,
  ];

  return { vec, ear: (A.ear + B.ear) / 2, head: h, eyes: { A, B } };
}

export const FEATURE_DIM = 24;
