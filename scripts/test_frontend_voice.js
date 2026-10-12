const assert = require('node:assert/strict');
const fs = require('node:fs');
const { VoiceTurnDetector, shouldSpeakReply } = require('../frontend/voice.js');

// Script startup must not abort before the mic handlers are registered due to
// missing voice-settings controls in the served HTML.
{
  const html = fs.readFileSync(require.resolve('../frontend/index.html'), 'utf8');
  for (const id of ['audio-settings', 'close-audio-settings', 'silence-duration', 'sensitivity-value',
    'voice-sensitivity', 'audio-input', 'audio-output', 'voice-output', 'audio-device-note']) {
    assert.match(html, new RegExp(`id=["']${id}["']`), `index.html must include #${id}`);
  }
  assert.match(html, /id="voice-sensitivity"[^>]*value="0\.03"/, 'the UI default must use the higher room-noise margin');
  assert.match(fs.readFileSync(require.resolve('../frontend/app.js'), 'utf8'), /tutor_vad_threshold_v5/,
    'a versioned key must discard previously saved lower defaults');
}

const rate = 48000;
const frameSize = 2048;
const frameSeconds = frameSize / rate;
let speechFrame = 0;
const speech = () => new Float32Array(frameSize).fill(speechFrame++ % 2 ? 0.09 : 0.05);
const silence = () => new Float32Array(frameSize);
assert.equal(shouldSpeakReply(), true);
assert.equal(shouldSpeakReply({ pendingBargeIn: true }), false);
assert.equal(shouldSpeakReply({ queuedTurns: 1 }), false);
assert.equal(shouldSpeakReply({ speechActive: true }), false);

function makeDetector(options = {}) {
  const turns = [];
  let starts = 0;
  const detector = new VoiceTurnDetector({
    sampleRate: rate,
    threshold: 0.02,
    silenceSeconds: 3,
    onSpeechStart: () => starts++,
    onUtterance: (samples, reason) => turns.push({ samples, reason }),
    ...options,
  });
  // The microphone's startup calibration uses a quiet room sample before the
  // synthetic speech, just as the live browser does.
  for (let i = 0; i < 20; i++) detector.push(silence(), frameSeconds);
  return { detector, turns, get starts() { return starts; } };
}

// A natural pause shorter than the endpoint is retained, and resumed speech
// resets the continuous-silence clock instead of splitting the sentence.
{
  const test = makeDetector();
  for (let i = 0; i < 20; i++) test.detector.push(speech(), frameSeconds);
  for (let i = 0; i < 60; i++) test.detector.push(silence(), frameSeconds);
  assert.equal(test.turns.length, 0);
  for (let i = 0; i < 10; i++) test.detector.push(speech(), frameSeconds);
  for (let i = 0; i < 71; i++) test.detector.push(silence(), frameSeconds);
  assert.equal(test.turns.length, 1);
  assert.equal(test.turns[0].reason, 'silence');
  assert.equal(test.starts, 1);
  for (let i = 0; i < 100; i++) test.detector.push(silence(), frameSeconds);
  assert.equal(test.turns.length, 1, 'silence must not create another empty turn');
}

// A manual microphone stop flushes the captured frames once, without waiting
// for the silence threshold; a repeated stop cannot submit them again.
{
  const test = makeDetector();
  for (let i = 0; i < 20; i++) test.detector.push(speech(), frameSeconds);
  assert.equal(test.detector.finalize('manual-stop'), true);
  assert.equal(test.turns.length, 1);
  assert.equal(test.turns[0].reason, 'manual-stop');
  assert.ok(test.turns[0].samples.length >= 20 * frameSize);
  assert.equal(test.detector.finalize('manual-stop'), false);
}

// Noise below the configured threshold and recordings shorter than 250 ms are
// ignored rather than becoming invalid LLM turns.
{
  const test = makeDetector();
  const noise = new Float32Array(frameSize).fill(0.005);
  for (let i = 0; i < 100; i++) test.detector.push(noise, frameSeconds);
  assert.equal(test.turns.length, 0);
  for (let i = 0; i < 5; i++) test.detector.push(speech(), frameSeconds);
  assert.equal(test.detector.finalize('manual-stop'), false);
  assert.equal(test.turns.length, 0);
}

// The buffer is bounded for extended utterances and resumes with a new turn.
{
  const test = makeDetector({ maxUtteranceSeconds: 1 });
  for (let i = 0; i < 30; i++) test.detector.push(speech(), frameSeconds);
  assert.equal(test.turns.length, 1);
  assert.equal(test.turns[0].reason, 'maximum-duration');
  assert.equal(test.detector.active, true, 'continued speech starts a fresh bounded segment');
  assert.ok(test.detector.frameCount <= test.detector.maxFrames);
  for (let i = 0; i < 30; i++) test.detector.push(speech(), frameSeconds);
  assert.equal(test.turns.length, 2);
  assert.equal(test.turns[1].reason, 'maximum-duration');
}

// The shipping default must calibrate steady room noise, distinguish a louder
// spoken turn, then treat that same room noise as silence.
{
  const turns = [];
  let lastLevel;
  const detector = new VoiceTurnDetector({
    sampleRate: rate,
    threshold: 0.004,
    silenceSeconds: 3,
    onLevel: level => { lastLevel = level; },
    onUtterance: (samples, reason) => turns.push({ samples, reason }),
  });
  const roomNoise = new Float32Array(frameSize).fill(0.052);
  const quietRoomNoise = new Float32Array(frameSize).fill(0.002);
  const speechFrames = [0.075, 0.095, 0.082, 0.11, 0.078, 0.10, 0.085, 0.12, 0.08, 0.10, 0.083, 0.11, 0.075, 0.095, 0.082, 0.11, 0.078, 0.10, 0.085, 0.12];
  for (let i = 0; i < 20; i++) detector.push(roomNoise, frameSeconds);
  assert.equal(detector.active, false, 'steady room noise above the absolute threshold must not open a turn');
  for (const level of speechFrames) detector.push(new Float32Array(frameSize).fill(level), frameSeconds);
  assert.equal(detector.active, true, 'speech louder than the calibrated noise floor must start a turn');
  for (let i = 0; i < 71; i++) detector.push(roomNoise, frameSeconds);
  assert.equal(turns.length, 1, 'three seconds of calibrated room noise must finish speech');
  assert.equal(turns[0].reason, 'silence');
  assert.ok(lastLevel && Number.isFinite(lastLevel.rms), 'input level is available for the live status');
  for (let i = 0; i < 80; i++) detector.push(quietRoomNoise, frameSeconds);
  assert.equal(turns.length, 1, 'silence after a completed turn must not create another turn');
}

// A low, steady room input near the level seen in the live browser must remain
// background after AGC settles; speech above that floor must still endpoint.
{
  const turns = [];
  const detector = new VoiceTurnDetector({
    sampleRate: rate,
    silenceSeconds: 3,
    onUtterance: (samples, reason) => turns.push({ samples, reason }),
  });
  const startupQuiet = new Float32Array(frameSize).fill(0.002);
  const roomNoise = new Float32Array(frameSize).fill(0.009);
  for (let i = 0; i < 4; i++) detector.push(startupQuiet, frameSeconds);
  for (let i = 0; i < 20; i++) detector.push(roomNoise, frameSeconds);
  for (let i = 0; i < 120; i++) detector.push(roomNoise, frameSeconds);
  assert.equal(detector.active, false, 'steady 0.009 RMS room noise must not open a turn');
  assert.equal(turns.length, 0, 'room noise must not be transcribed as a user turn');
  for (let i = 0; i < 16; i++) detector.push(new Float32Array(frameSize).fill(i % 2 ? 0.075 : 0.06), frameSeconds);
  assert.equal(detector.active, true, 'speech above the live room floor must open a turn');
  for (let i = 0; i < 71; i++) detector.push(roomNoise, frameSeconds);
  assert.equal(turns.length, 1);
  assert.equal(turns[0].reason, 'silence');
}

// The live browser also reported a steady 0.006 RMS floor. The default
// sensitivity must keep that level below speech onset after calibration.
{
  const turns = [];
  const detector = new VoiceTurnDetector({
    sampleRate: rate,
    silenceSeconds: 3,
    onUtterance: (samples, reason) => turns.push({ samples, reason }),
  });
  const roomNoise = new Float32Array(frameSize).fill(0.006);
  for (let i = 0; i < 120; i++) detector.push(roomNoise, frameSeconds);
  assert.equal(detector.active, false, 'steady 0.006 RMS room noise must not open a turn');
  assert.equal(turns.length, 0);
}

// A brief sound above the room threshold is not a spoken turn and must not
// leave the UI in a false active/listening state.
{
  const turns = [];
  const detector = new VoiceTurnDetector({
    sampleRate: rate,
    silenceSeconds: 3,
    onUtterance: (samples, reason) => turns.push({ samples, reason }),
  });
  for (let i = 0; i < 20; i++) detector.push(silence(), frameSeconds);
  const transient = new Float32Array(frameSize).fill(0.04);
  for (let i = 0; i < 4; i++) detector.push(transient, frameSeconds);
  for (let i = 0; i < 120; i++) detector.push(silence(), frameSeconds);
  assert.equal(detector.active, false, 'short transient must not become active speech');
  assert.equal(turns.length, 0, 'short transient must not be sent for transcription');
}

// The user's live meter showed noise spikes over a 0.003 floor. The higher
// shipping default rejects low and sustained near-speech noise while preserving
// stronger speech.
{
  const turns = [];
  const detector = new VoiceTurnDetector({
    sampleRate: rate,
    silenceSeconds: 3,
    onUtterance: (samples, reason) => turns.push({ samples, reason }),
  });
  const floor = new Float32Array(frameSize).fill(0.003);
  const spike = new Float32Array(frameSize).fill(0.026);
  const steadyNoise = new Float32Array(frameSize).fill(0.04);
  for (let i = 20; i-- > 0;) detector.push(floor, frameSeconds);
  for (let i = 0; i < 8; i++) detector.push(spike, frameSeconds);
  assert.equal(detector.active, false, 'a low room-noise spike must stay below the shipping default');
  for (let i = 0; i < 40; i++) detector.push(steadyNoise, frameSeconds);
  assert.equal(detector.active, false, 'steady below-trigger ambient noise must not open a turn');
  assert.equal(turns.length, 0);
  for (let i = 0; i < 10; i++) detector.push(new Float32Array(frameSize).fill(0.12), frameSeconds);
  assert.equal(detector.active, true, 'clear speech must cross the higher default threshold');
}

console.log('9 frontend voice lifecycle checks passed');
