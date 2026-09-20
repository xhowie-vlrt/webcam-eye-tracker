// Demo UI. All of the actual tracking lives in EyeTracker; this file is only
// wiring, so it doubles as a worked example of the public API.

import { EyeTracker, pixelsToDegrees } from './eyetracker.js';
import { OVERLAY_POINTS } from './features.js';
import { Heatmap } from './heatmap.js';
import { ScanPath } from './scanpath.js';

const $ = (id) => document.getElementById(id);

const el = Object.fromEntries(
  [
    'video', 'mesh', 'previewHint', 'status', 'gazeDot', 'heatmap', 'scanpath',
    'btnStart', 'btnCalibrate', 'btnValidate', 'btnRecord', 'btnExport', 'btnReset',
    'chkDot', 'chkHeatmap', 'chkScanpath', 'chkMesh', 'chkDrift',
    'calMode', 'smoothing', 'smoothingValue',
    'statFps', 'statSamples', 'statCv', 'statValidation', 'statFixations', 'statLog',
  ].map((id) => [id, $(id)])
);

const tracker = new EyeTracker({ video: el.video });
const heatmap = new Heatmap(el.heatmap);
const scanpath = new ScanPath(el.scanpath);
const meshCtx = el.mesh.getContext('2d');

const state = { recording: false, log: [], fixations: 0, busy: false, lastHeat: 0 };

// ------------------------------------------------------------------ status --

function setStatus(text, kind = '') {
  el.status.textContent = text;
  el.status.className = `status ${kind}`.trim();
}

function refresh() {
  const live = tracker.running;
  el.btnStart.textContent = live ? 'カメラを停止' : 'カメラを開始';
  el.btnCalibrate.disabled = !live || state.busy;
  el.btnValidate.disabled = !live || !tracker.calibrated || state.busy;
  el.btnRecord.disabled = !tracker.calibrated || state.busy;
  el.btnExport.disabled = state.log.length === 0;
  el.btnReset.disabled = !tracker.calibrated;
  el.btnRecord.textContent = state.recording ? '記録を停止' : '記録を開始';
  el.btnRecord.classList.toggle('active', state.recording);
  el.statSamples.textContent = String(tracker.model.sampleCount);
  el.statLog.textContent = String(state.log.length);
  el.statFixations.textContent = String(state.fixations);
}

const fmt = (px) => `${px.toFixed(0)} px (≈${pixelsToDegrees(px).toFixed(1)}°)`;

// ------------------------------------------------------------------ events --

tracker.on('face', ({ found, landmarks, fps }) => {
  el.statFps.textContent = fps ? fps.toFixed(0) : '-';
  el.previewHint.classList.toggle('hidden', found);
  if (!found) {
    el.previewHint.textContent = '顔が検出できません';
    meshCtx.clearRect(0, 0, el.mesh.width, el.mesh.height);
    return;
  }
  drawMesh(landmarks);
});

tracker.on('gaze', ({ x, y, blink, timestamp }) => {
  el.gazeDot.style.transform = `translate(${x}px, ${y}px)`;
  el.gazeDot.classList.toggle('hidden', !el.chkDot.checked);
  el.gazeDot.classList.toggle('blink', !!blink);
  if (blink) return;

  if (el.chkHeatmap.checked) {
    heatmap.add(x, y);
    if (timestamp - state.lastHeat > 120) {
      state.lastHeat = timestamp;
      heatmap.render();
    }
  }
  if (el.chkScanpath.checked) scanpath.addGaze(x, y);
  if (state.recording) {
    state.log.push({ t: Math.round(timestamp), x, y });
    if (state.log.length % 15 === 0) refresh();
  }
});

tracker.on('fixation', (f) => {
  state.fixations++;
  if (el.chkScanpath.checked) scanpath.addFixation(f);
  el.statFixations.textContent = String(state.fixations);
});

tracker.on('quality', ({ ok, reason }) => {
  el.gazeDot.classList.toggle('degraded', !ok);
  setStatus(
    ok
      ? 'キャリブレーション時の姿勢に戻りました'
      : `${reason}がキャリブレーション時と違います — 精度が落ちています`,
    ok ? 'ok' : 'warn'
  );
});

tracker.on('error', (err) => setStatus(describe(err), 'error'));

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

// ----------------------------------------------------------------- actions --

el.btnStart.addEventListener('click', async () => {
  if (tracker.running) {
    tracker.stop();
    el.gazeDot.classList.add('hidden');
    el.previewHint.textContent = 'カメラ未接続';
    el.previewHint.classList.remove('hidden');
    setStatus('カメラは停止中です');
    refresh();
    return;
  }
  await withBusy('モデルを読み込み中… (初回のみ数秒)', async () => {
    await tracker.start();
    setStatus(
      tracker.calibrated
        ? '保存済みモデルで推定中。ずれていたら再キャリブレーションを。'
        : 'カメラ稼働中。まずキャリブレーションしてください。',
      'ok'
    );
  });
});

const MODES = {
  click: { mode: 'click' },
  pursuit: { mode: 'pursuit' },
  both: { mode: 'both' },
  click16: { mode: 'click', cols: 4, rows: 4 },
};

el.btnCalibrate.addEventListener('click', async () => {
  await withBusy('キャリブレーション中…', async () => {
    const report = await tracker.calibrate(MODES[el.calMode.value]);
    if (!report) {
      setStatus('キャリブレーションを中止しました', 'warn');
      return;
    }
    heatmap.clear();
    scanpath.clear();
    state.fixations = 0;
    el.statCv.textContent = report.cv == null ? '-' : fmt(report.cv);
    el.statValidation.textContent = '-';
    setStatus(
      `完了: ${report.samples} サンプル / ${report.groups} 点、` +
        `交差検証誤差 ${report.cv == null ? '-' : fmt(report.cv)}`,
      'ok'
    );
  });
});

el.btnValidate.addEventListener('click', async () => {
  await withBusy('精度チェック中…', async () => {
    const r = await tracker.validate();
    if (!r) {
      setStatus('精度チェックを中止しました', 'warn');
      return;
    }
    el.statValidation.textContent = fmt(r.mean);
    setStatus(
      `平均 ${fmt(r.mean)}、最大 ${r.max.toFixed(0)} px — ` +
        (r.degrees < 1.5
          ? 'この機材ではかなり良好です'
          : r.degrees < 3
            ? 'Webカメラとしては標準的です'
            : '照明・姿勢を見直して再キャリブレーションを'),
      r.degrees < 3 ? 'ok' : 'warn'
    );
  });
});

el.btnRecord.addEventListener('click', () => {
  state.recording = !state.recording;
  if (state.recording) {
    state.log = [];
    setStatus('視線を記録中…', 'busy');
  } else {
    setStatus(`記録を停止しました (${state.log.length} 点)`, 'ok');
  }
  refresh();
});

el.btnExport.addEventListener('click', () => {
  const rows = state.log.map((r) => `${r.t},${r.x.toFixed(1)},${r.y.toFixed(1)}`);
  download(
    `timestamp_ms,x_px,y_px\n${rows.join('\n')}\n`,
    `gaze-${new Date().toISOString().replace(/[:.]/g, '-')}.csv`
  );
});

el.btnReset.addEventListener('click', () => {
  tracker.clearModel();
  heatmap.clear();
  scanpath.clear();
  state.recording = false;
  state.fixations = 0;
  el.gazeDot.classList.add('hidden');
  el.statCv.textContent = '-';
  el.statValidation.textContent = '-';
  setStatus('モデルを消去しました', 'warn');
  refresh();
});

el.chkHeatmap.addEventListener('change', () => {
  if (!el.chkHeatmap.checked) heatmap.clear();
});
el.chkScanpath.addEventListener('change', () => {
  if (!el.chkScanpath.checked) scanpath.clear();
});
el.chkDot.addEventListener('change', () => {
  el.gazeDot.classList.toggle('hidden', !el.chkDot.checked);
});
el.chkDrift.addEventListener('change', () => {
  tracker.setDriftCorrection(el.chkDrift.checked);
  if (el.chkDrift.checked) {
    setStatus('ドリフト補正オン: クリックした位置を見ていたものとして学習します', 'ok');
  }
});

el.smoothing.addEventListener('input', () => {
  el.smoothingValue.textContent = el.smoothing.value;
  tracker.setSmoothing(Number(el.smoothing.value) / 100);
});

window.addEventListener('resize', () => {
  heatmap.resize();
  scanpath.resize();
});

// ------------------------------------------------------------------- utils --

async function withBusy(message, fn) {
  state.busy = true;
  refresh();
  setStatus(message, 'busy');
  try {
    await fn();
  } catch (err) {
    console.error(err);
    setStatus(describe(err), 'error');
  } finally {
    state.busy = false;
    refresh();
  }
}

function download(text, filename) {
  const url = URL.createObjectURL(new Blob([text], { type: 'text/csv' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function describe(err) {
  if (err?.name === 'NotAllowedError') return 'カメラの使用が許可されていません';
  if (err?.name === 'NotFoundError') return 'カメラが見つかりません';
  if (err?.name === 'NotReadableError') return '他のアプリがカメラを使用中です';
  return err?.message ?? String(err);
}

// -------------------------------------------------------------------- boot --

if (!navigator.mediaDevices?.getUserMedia) {
  setStatus('このブラウザは getUserMedia に対応していません', 'error');
  el.btnStart.disabled = true;
} else if (!window.isSecureContext) {
  setStatus('https か localhost で開いてください（カメラは安全な文脈でのみ使えます）', 'error');
  el.btnStart.disabled = true;
} else if (tracker.loadModel()) {
  setStatus('保存済みのキャリブレーションを読み込みました。カメラを開始してください。');
}

tracker.setSmoothing(Number(el.smoothing.value) / 100);
refresh();

if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('sw.js').catch(() => {
    /* offline support is a bonus, not a requirement */
  });
}
