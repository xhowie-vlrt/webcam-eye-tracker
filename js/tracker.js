// Camera capture + MediaPipe FaceLandmarker, wrapped in a small event emitter.

import { resolveAssets } from './config.js';

export class FaceTracker {
  constructor(video) {
    this.video = video;
    this.landmarker = null;
    this.stream = null;
    this.running = false;
    this.lastTimestamp = -1;
    this.onResult = () => {};
    this.fps = 0;
    this._frameTimes = [];
  }

  async load() {
    if (this.landmarker) return;
    const assets = resolveAssets();
    this.assets = assets;
    // Loaded dynamically so the source of the runtime stays configurable; see
    // js/config.js.
    const { FaceLandmarker, FilesetResolver } = await import(
      /* @vite-ignore */ assets.bundle
    );
    const fileset = await FilesetResolver.forVisionTasks(assets.wasm);
    this.landmarker = await FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: assets.model, delegate: 'GPU' },
      runningMode: 'VIDEO',
      numFaces: 1,
      // The model bundle already returns the 10 iris points we depend on.
      outputFaceBlendshapes: false,
      outputFacialTransformationMatrixes: false,
    });
  }

  async start() {
    await this.load();
    if (this.running) return;

    this.stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 30 },
        facingMode: 'user',
      },
      audio: false,
    });
    this.video.srcObject = this.stream;
    await this.video.play();

    this.running = true;
    this._loop();
  }

  stop() {
    this.running = false;
    if (this.stream) {
      for (const track of this.stream.getTracks()) track.stop();
      this.stream = null;
    }
    this.video.srcObject = null;
  }

  _schedule(fn) {
    // requestVideoFrameCallback fires once per decoded camera frame, which
    // avoids running the model twice on the same image on a 120 Hz display.
    if (this.video.requestVideoFrameCallback) {
      this.video.requestVideoFrameCallback(() => fn());
    } else {
      requestAnimationFrame(() => fn());
    }
  }

  _loop() {
    if (!this.running) return;

    const now = performance.now();
    if (this.video.readyState >= 2 && now > this.lastTimestamp) {
      this.lastTimestamp = now;
      let result = null;
      try {
        result = this.landmarker.detectForVideo(this.video, now);
      } catch (err) {
        console.warn('detectForVideo failed', err);
      }
      if (result) {
        this._trackFps(now);
        const landmarks = result.faceLandmarks?.[0] ?? null;
        this.onResult({ landmarks, timestamp: now });
      }
    }
    this._schedule(() => this._loop());
  }

  _trackFps(now) {
    this._frameTimes.push(now);
    while (this._frameTimes.length > 30) this._frameTimes.shift();
    const span = this._frameTimes.at(-1) - this._frameTimes[0];
    if (span > 0) this.fps = ((this._frameTimes.length - 1) * 1000) / span;
  }
}
