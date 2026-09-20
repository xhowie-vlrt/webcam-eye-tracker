// Where the MediaPipe wasm runtime and the face-landmarker model come from.
//
// By default they are fetched from jsDelivr and Google's model host, which
// makes the repo small and `git clone && npm start` work immediately. Run
// `npm run setup` to download them into ./vendor instead: the app then makes
// no third-party requests at all, works offline, and cannot break because a
// CDN did.
//
// Resolution order:
//   1. globalThis.EYETRACKER_ASSETS  - for embedders who bundle their own copy
//   2. <meta name="eyetracker-assets" content="vendor">  - written by npm run setup
//   3. the CDN

const CDN = {
  bundle: 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs',
  wasm: 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm',
  model:
    'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
};

/** @returns {{bundle:string, wasm:string, model:string, local:boolean}} */
export function resolveAssets() {
  const override =
    globalThis.EYETRACKER_ASSETS ??
    document.querySelector('meta[name="eyetracker-assets"]')?.content ??
    null;

  if (!override) return { ...CDN, local: false };

  if (typeof override === 'object') return { ...CDN, ...override, local: true };

  // A string is a base directory, resolved relative to the page.
  const base = new URL(
    override.endsWith('/') ? override : `${override}/`,
    document.baseURI
  ).href;
  return {
    bundle: `${base}tasks-vision/vision_bundle.mjs`,
    wasm: `${base}tasks-vision/wasm`,
    model: `${base}models/face_landmarker.task`,
    local: true,
  };
}

export const CDN_ASSETS = CDN;
