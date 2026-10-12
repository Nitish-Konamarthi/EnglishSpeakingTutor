/* Small, deterministic browser-side voice activity and endpoint detector. */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.VoiceCapture = api;
})(globalThis, function () {
  class VoiceTurnDetector {
    constructor({ threshold = 0.03, silenceSeconds = 3, sampleRate = 48000,
      preRollSeconds = 0.75, maxUtteranceSeconds = 90, onSpeechStart = () => {},
      onUtterance = () => {}, onLevel = () => {} } = {}) {
      this.threshold = threshold;
      this.silenceSeconds = silenceSeconds;
      this.sampleRate = sampleRate;
      this.preRollFrames = Math.ceil(preRollSeconds * sampleRate / 2048);
      this.maxFrames = Math.ceil(maxUtteranceSeconds * sampleRate / 2048);
      this.onSpeechStart = onSpeechStart;
      this.onUtterance = onUtterance;
      this.onLevel = onLevel;
      this.noiseFloor = 0;
      this.calibrationLevels = [];
      this.calibrationElapsed = 0;
      this.calibrated = false;
      this.onsetCandidate = [];
      this.speechPeak = 0;
      this.reset(false);
    }

    reset(retainCalibration = true) {
      this.preRoll = [];
      this.frames = [];
      this.active = false;
      this.silence = 0;
      this.frameCount = 0;
      this.voicedSamples = 0;
      this.speechPeak = 0;
      this.onsetCandidate = [];
      if (!retainCalibration) {
        this.noiseFloor = 0;
        this.calibrationLevels = [];
        this.calibrationElapsed = 0;
        this.calibrated = false;
      }
    }

    push(frame, elapsedSeconds) {
      const copy = new Float32Array(frame);
      const rms = Math.sqrt(copy.reduce((sum, value) => sum + value * value, 0) / Math.max(1, copy.length));
      if (!this.active) {
        this.preRoll.push(copy);
        if (this.preRoll.length > this.preRollFrames) this.preRoll.shift();
        if (!this.calibrated) {
          this.calibrationLevels.push(rms);
          this.calibrationElapsed += Math.max(0, elapsedSeconds);
          // Give browser AGC/noise suppression time to settle before choosing
          // the room floor. The pre-roll keeps early speech frames available.
          if (this.calibrationElapsed < 0.75) {
            this.onLevel({ rms, voiced: false, active: false, calibrating: true, silenceSeconds: 0,
              endpointSeconds: this.silenceSeconds, threshold: this.threshold });
            return;
          }
          const sorted = [...this.calibrationLevels].sort((a, b) => a - b);
          this.noiseFloor = sorted[Math.floor((sorted.length - 1) * 0.7)] || 0;
          this.calibrated = true;
          this.calibrationLevels = [];
          this.preRollFrames = Math.ceil(0.25 * this.sampleRate / 2048);
          while (this.preRoll.length > this.preRollFrames) this.preRoll.shift();
        }
        // Apply a configurable margin to the measured room floor. Require a
        // quarter second of sustained activity to reject short noise spikes.
        const startThreshold = Math.max(this.threshold * 1.5, this.noiseFloor * (1 + this.threshold * 120));
        if (rms < startThreshold) {
          this.onsetCandidate = [];
          this.noiseFloor = this.noiseFloor * 0.98 + rms * 0.02;
          this.onLevel({ rms, voiced: false, active: false, calibrating: false, silenceSeconds: 0,
            endpointSeconds: this.silenceSeconds, threshold: startThreshold, noiseFloor: this.noiseFloor });
          return;
        }
        this.onsetCandidate.push(copy);
        if (this.onsetCandidate.length < 6) {
          this.onLevel({ rms, voiced: false, active: false, calibrating: false, silenceSeconds: 0,
            endpointSeconds: this.silenceSeconds, threshold: startThreshold, noiseFloor: this.noiseFloor });
          return;
        }
        this.active = true;
        this.frames = this.preRoll;
        this.preRoll = [];
        this.frameCount = this.frames.length;
        this.voicedSamples = (this.onsetCandidate.length - 1) * copy.length;
        this.onsetCandidate = [];
        this.speechPeak = rms;
        this.onSpeechStart();
      } else {
        this.frames.push(copy);
        this.frameCount += 1;
        if (rms > this.speechPeak) this.speechPeak = rms;
      }

      const silenceThreshold = Math.max(this.threshold * 0.6, this.noiseFloor * (1 + this.threshold * 120));
      const voiced = rms >= silenceThreshold;
      this.onLevel({ rms, voiced, active: this.active, calibrating: false, silenceSeconds: this.silence,
        endpointSeconds: this.silenceSeconds, threshold: silenceThreshold, noiseFloor: this.noiseFloor });
      if (voiced) this.voicedSamples += copy.length;
      else if (this.speechPeak > 0 && rms < this.speechPeak * 0.35) {
        this.noiseFloor = this.noiseFloor * 0.98 + rms * 0.02;
      }
      this.silence = voiced ? 0 : this.silence + Math.max(0, elapsedSeconds);
      if (this.silence >= this.silenceSeconds || this.frameCount >= this.maxFrames) {
        const completed = this.finalize(this.silence >= this.silenceSeconds ? 'silence' : 'maximum-duration');
        if (!completed) this.onLevel({ rms, voiced: false, active: false, calibrating: false, silenceSeconds: 0,
          endpointSeconds: this.silenceSeconds, threshold: silenceThreshold, noiseFloor: this.noiseFloor });
      }
    }

    finalize(reason = 'manual-stop') {
      if (!this.active) return false;
      const frames = this.frames;
      const voicedSamples = this.voicedSamples;
      this.reset(true);
      const sampleCount = frames.reduce((total, frame) => total + frame.length, 0);
      if (voicedSamples < this.sampleRate * 0.25) return false;
      const samples = new Float32Array(sampleCount);
      let offset = 0;
      for (const frame of frames) {
        samples.set(frame, offset);
        offset += frame.length;
      }
      this.onUtterance(samples, reason);
      return true;
    }
  }

  function shouldSpeakReply({ pendingBargeIn = false, queuedTurns = 0, speechActive = false } = {}) {
    return !pendingBargeIn && queuedTurns === 0 && !speechActive;
  }

  return { VoiceTurnDetector, shouldSpeakReply };
});
