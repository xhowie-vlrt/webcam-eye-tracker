// Wiring: camera -> landmarks -> features -> gaze model -> screen.

import { FaceTracker } from './tracker.js';
import { buildFeatures, OVERLAY_POINTS } from './features.js';
import { GazeModel, pixelsToDegrees } from './gaze.js';
import {
  CalibrationRunner,
  gridPoints,
  validationPoints,
  shuffle,
} from './calibration.js';
import { Heatmap } from './heatmap.js';
import { OneEuroPoint } from './filter.js';

const BLINK_EAR = 0.17; // eye-aspect-ratio below this = lid closed
const HEATMAP_INTERVAL_MS = 120;

const $ = (id) => document.getElementById(id);

const el = {
  video: $('video'),
  mesh: $('mesh'),
  previewHint: $('previewHint'),
  status: $('status'),
  gazeDot: $('gazeDot'),
  heatmap: $('heatmap'),
  btnStart: $('btnStart'),
  btnCalibrate: $('btnCalibrate'),
  btnValidate: $('btnValidate'),
  btnRecord: $('btnRecord'),
  btnExport: $('btnExport'),
  btnReset: $('btnReset'),
  chkDot: $('chkDot'),
  chkHeatmap: $('chkHeatmap'),
  chkMesh: $('chkMesh'),
  smoothing: $('smoothing'),
  smoothingValue: $('smoothingValue'),
  statFps: $('statFps'),
  statSamples: $('statSamples'),
  statTrain: $('statTrain'),
  statValidation: $('statValidation'),
  statBlink: $('statBlink'),
  statLog: $('statLog'),
  calOverlay: $('calibration'),
  calDot: $('calDot'),
  calHint: $('calHint'),
};

const tracker = new FaceTracker(el.video);
const model = new GazeModel();
const heatmap = new Heatmap(el.heatmap);
const smoother = new OneEuroPoint({ minCutoff: 0.6, beta: 0.012 });
const meshCtx = el.mesh.getContext('2d');

const state = {
  latest: null, // { vec, ear, timestamp }
  blinking: false,
  recording: false,
  log: [],
  lastHeatmapRender: 0,
  busy: false,
};

const calibration = new CalibrationRunner({
  overlay: el.calOverlay,
  dot: el.calDot,
  hint: el.calHint,
  // Blinks would poison the training set, so refuse to hand one out.
  getSample: () =>
    state.latest && !state.blinking ? state.latest.vec : null,
});

// ---------------------------------------------------------------- status ---

function setStatus(text, kind = '') {
  el.status.textContent = text;
  el.status.className = `status ${kind}`.trim();
}

function refreshButtons() {
  const live = tracker.running;
  el.btnStart.textContent = live ? 'カメラを停止' : 'カメラを開始';
  el.btnCalibrate.disabled = !live || state.busy;
  el.btnValidate.disabled = !live || !model.ready || state.busy;
  el.btnRecord.disabled = !model.ready || state.busy;
  el.btnExport.disabled = state.log.length === 0;
  el.btnReset.disabled = !model.ready && model.sampleCount === 0;
  el.btnRecord.textContent = state.recording ? '記録を停止' : '記録を開始';
  el.btnRecord.classList.toggle('active', state.recording);
  el.statSamples.textContent = String(model.sampleCount);
  el.statLog.textContent = String(state.log.length);
}

function formatError(err) {
  if (!err) return '-';
  const deg = pixelsToDegrees(err.mean);
  return `${err.mean.toFixed(0)} px (≈${deg.toFixed(1)}°)`;
}

// ----------------------------------------------------------------- frames ---

tracker.onResult = ({ landmarks, timestamp }) => {
  el.statFps.textContent = tracker.fps ? tracker.fps.toFixed(0) : '-';

  if (!landmarks) {
    state.latest = null;
    el.previewHint.textContent = '顔が検出できません';
    el.previewHint.classList.remove('hidden');
    meshCtx.clearRect(0, 0, el.mesh.width, el.mesh.height);
    return;
  }
  el.previewHint.classList.add('hidden');

  const aspect = (el.video.videoWidth || 16) / (el.video.videoHeight || 9);
  const f = buildFeatures(landmarks, aspect);

  state.blinking = f.ear < BLINK_EAR;
  el.statBlink.textContent = state.blinking
    ? '閉じている'
    : f.ear.toFixed(2);
  state.latest = { vec: f.vec, ear: f.ear, timestamp };

  drawMesh(landmarks);

  if (model.ready && !state.blinking) {
    const raw = model.predict(f.vec);
    const s = smoother.filter(raw.x, raw.y, timestamp / 1000);
    const x = clamp(s.x, 0, window.innerWidth);
    const y = clamp(s.y, 0, window.innerHeight);
    el.gazeDot.style.transform = `translate(${x}px, ${y}px)`;
    el.gazeDot.classList.toggle('hidden', !el.chkDot.checked);
    el.gazeDot.classList.remove('blink');

    if (el.chkHeatmap.checked) {
      heatmap.add(x, y);
      if (timestamp - state.lastHeatmapRender > HEATMAP_INTERVAL_MS) {
        state.lastHeatmapRender = timestamp;
        heatmap.render();
      }
    }
    if (state.recording) {
      state.log.push({ t: Math.round(timestamp), x, y, ear: f.ear });
      if (state.log.length % 15 === 0) refreshButtons();
    }
  } else if (model.ready) {
    el.gazeDot.classList.add('blink');
  }
};

function drawMesh(landmarks) {
  const w = el.video.videoWidth;
  const h = el.video.videoHeight;
  if (!w || !h) return;
  if (el.mesh.width !== w || el.mesh.height !== h) {
    el.mesh.width = w;
    el.mesh.height = h;
  }
  meshCtx.clearRect(0, 0, w, h);
  if (!el.chkMesh.checked) return;

  meshCtx.fillStyle = '#5ac8fa';
  for (const i of OVERLAY_POINTS) {
    const p = landmarks[i];
    if (!p) continue;
    meshCtx.beginPath();
    meshCtx.arc(p.x * w, p.y * h, 3, 0, Math.PI * 2);
    meshCtx.fill();
  }
}

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

// ----------------------------------------------------------------- actions ---

el.btnStart.addEventListener('click', async () => {
  if (tracker.running) {
    tracker.stop();
    state.latest = null;
    el.gazeDot.classList.add('hidden');
    el.previewHint.textContent = 'カメラ未接続';
    el.previewHint.classList.remove('hidden');
    setStatus('カメラは停止中です');
    refreshButtons();
    return;
  }

  state.busy = true;
  refreshButtons();
  setStatus('モデルを読み込み中… (初回は数秒かかります)', 'busy');
  try {
    await tracker.start();
    setStatus(
      model.ready
        ? '保存済みモデルで推定中。ずれていたら再キャリブレーションしてください。'
        : 'カメラ稼働中。キャリブレーションを実行してください。',
      'ok'
    );
    el.previewHint.classList.add('hidden');
  } catch (err) {
    console.error(err);
    setStatus(`カメラを開始できません: ${describe(err)}`, 'error');
  } finally {
    state.busy = false;
    refreshButtons();
  }
});

el.btnCalibrate.addEventListener('click', async () => {
  state.busy = true;
  refreshButtons();
  const points = shuffle(gridPoints(3, 3));
  const collected = await calibration.run(points, 'キャリブレーション');

  if (collected.length < points.length) {
    setStatus('キャリブレーションを中止しました', 'warn');
    state.busy = false;
    refreshButtons();
    return;
  }

  model.clear();
  for (const { x, y, vecs } of collected) {
    for (const vec of vecs) model.addSample(vec, x, y);
  }

  try {
    const err = model.fit();
    model.save();
    smoother.reset();
    el.statTrain.textContent = formatError(err);
    el.statValidation.textContent = '-';
    setStatus(`キャリブレーション完了 (学習誤差 ${formatError(err)})`, 'ok');
  } catch (err) {
    console.error(err);
    setStatus(`学習に失敗しました: ${describe(err)}`, 'error');
  } finally {
    state.busy = false;
    refreshButtons();
  }
});

el.btnValidate.addEventListener('click', async () => {
  state.busy = true;
  refreshButtons();
  const points = shuffle(validationPoints());
  const collected = await calibration.run(points, '精度チェック');

  if (collected.length < points.length) {
    setStatus('精度チェックを中止しました', 'warn');
    state.busy = false;
    refreshButtons();
    return;
  }

  // Average the prediction over each fixation, then measure that against the
  // target - this is the number that matches how the tracker is actually used.
  let total = 0;
  let worst = 0;
  for (const { x, y, vecs } of collected) {
    let sx = 0;
    let sy = 0;
    for (const vec of vecs) {
      const p = model.predict(vec);
      sx += p.x;
      sy += p.y;
    }
    const e = Math.hypot(sx / vecs.length - x, sy / vecs.length - y);
    total += e;
    worst = Math.max(worst, e);
  }
  const mean = total / collected.length;
  el.statValidation.textContent = formatError({ mean });
  setStatus(
    `平均誤差 ${mean.toFixed(0)} px (≈${pixelsToDegrees(mean).toFixed(1)}°)、` +
      `最大 ${worst.toFixed(0)} px`,
    mean < 150 ? 'ok' : 'warn'
  );
  state.busy = false;
  refreshButtons();
});

el.btnRecord.addEventListener('click', () => {
  state.recording = !state.recording;
  if (state.recording) {
    state.log = [];
    setStatus('視線を記録中…', 'busy');
  } else {
    setStatus(`記録を停止しました (${state.log.length} 点)`, 'ok');
  }
  refreshButtons();
});

el.btnExport.addEventListener('click', () => {
  const header = 'timestamp_ms,x_px,y_px,eye_aspect_ratio,screen_w,screen_h';
  const w = window.innerWidth;
  const h = window.innerHeight;
  const rows = state.log.map(
    (r) => `${r.t},${r.x.toFixed(1)},${r.y.toFixed(1)},${r.ear.toFixed(4)},${w},${h}`
  );
  const blob = new Blob([`${header}\n${rows.join('\n')}\n`], {
    type: 'text/csv',
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `gaze-${new Date().toISOString().replace(/[:.]/g, '-')}.csv`;
  a.click();
  URL.revokeObjectURL(url);
});

el.btnReset.addEventListener('click', () => {
  model.clear();
  GazeModel.forget();
  smoother.reset();
  heatmap.clear();
  state.recording = false;
  el.gazeDot.classList.add('hidden');
  el.statTrain.textContent = '-';
  el.statValidation.textContent = '-';
  setStatus('モデルを消去しました', 'warn');
  refreshButtons();
});

el.chkHeatmap.addEventListener('change', () => {
  if (!el.chkHeatmap.checked) heatmap.clear();
});

el.chkDot.addEventListener('change', () => {
  el.gazeDot.classList.toggle('hidden', !el.chkDot.checked);
});

el.smoothing.addEventListener('input', () => {
  const v = Number(el.smoothing.value);
  el.smoothingValue.textContent = String(v);
  // 0 -> responsive but jumpy, 100 -> very smooth but laggy.
  const minCutoff = 8 * Math.pow(0.02 / 8, v / 100);
  smoother.setParams({ minCutoff, beta: 0.02 });
});
el.smoothing.dispatchEvent(new Event('input'));

window.addEventListener('resize', () => heatmap.resize());

function describe(err) {
  if (err?.name === 'NotAllowedError') return 'カメラの使用が許可されていません';
  if (err?.name === 'NotFoundError') return 'カメラが見つかりません';
  if (err?.name === 'NotReadableError') return '他のアプリがカメラを使用中です';
  return err?.message ?? String(err);
}

// ------------------------------------------------------------------- boot ---

if (!navigator.mediaDevices?.getUserMedia) {
  setStatus('このブラウザは getUserMedia に対応していません', 'error');
  el.btnStart.disabled = true;
} else if (!window.isSecureContext) {
  setStatus('https か localhost で開いてください (カメラは安全な文脈でのみ使えます)', 'error');
  el.btnStart.disabled = true;
} else if (model.load()) {
  el.statTrain.textContent = formatError(model.trainError);
  setStatus('保存済みのキャリブレーションを読み込みました。カメラを開始してください。');
}

refreshButtons();
