const $ = (q) => document.querySelector(q);
const state = { session: null, busy: false, textTurn: false, voiceQueueRunning: false, pendingSocketTurn: null, pendingBargeIn: false, stream: null, audioContext: null, processor: null, detector: null, startingMic: false, micRequest: 0, lastAudioFrameAt: 0, lastLevelStatusAt: 0, captureWatchdog: null, player: null, playerUrl: null, playbackController: null, playbackId: 0, playbackQueue: [], playbackSpaceResolvers: [], playbackDrainResolvers: [], voiceQueue: [], turnEpoch: 0 };
const status = (label, mode='') => { $('#status-text').textContent = label; $('#status').className = `status ${mode}`; };
async function api(path, options={}) { const response = await fetch(path, options); const data = await response.json().catch(()=>({})); if(!response.ok) throw new Error(data.detail || `Request failed (${response.status})`); return data; }
function bubble(role, text) { $('#welcome').hidden = true; const row=document.createElement('div'); row.className=`message ${role}`; const avatar=document.createElement('div'); avatar.className='avatar'; avatar.textContent=role==='assistant'?'e':'YOU'; const body=document.createElement('div'); body.className='bubble'; body.textContent=text; row.append(avatar,body); $('#messages').append(row); $('#messages').scrollTop=$('#messages').scrollHeight; return row; }
function renderMessages(messages) { $('#messages').replaceChildren(); if(!messages.length) $('#welcome').hidden=false; else { $('#welcome').hidden=true; messages.forEach(m=>bubble(m.role,m.content)); } }
async function refreshSessions() { const {sessions}=await api('/api/v1/sessions'); const nav=$('#sessions'); nav.replaceChildren(); sessions.forEach(s=>{const b=document.createElement('button');b.className=`session-item ${s.id===state.session?'active':''}`;b.textContent=s.title;b.onclick=()=>openSession(s.id);nav.append(b);}); }
async function openSession(id) {
 if(state.busy||state.stream||state.startingMic){status('Stop voice practice and finish the current turn before changing conversations.','error');return;}
 if(matchMedia('(max-width:700px)').matches){$('.sidebar').classList.remove('open');$('#menu-toggle').setAttribute('aria-expanded','false');}
 state.turnEpoch++;
 interruptPlayback();
 if(state.socket)state.socket.close();state.session=id;
 const data=await api(`/api/v1/sessions/${id}`);renderMessages(data.messages);await refreshSessions();
 const scheme=location.protocol==='https:'?'wss':'ws',socket=new WebSocket(`${scheme}://${location.host}/api/v1/ws/${id}`);state.socket=socket;
 socket.onopen=()=>{if(state.socket===socket)status('Ready when you are');};
 socket.onmessage=e=>{if(state.socket!==socket)return;const event=JSON.parse(e.data);if(event.type==='transcript')bubble('user',event.text);else if(event.type==='assistant_text'){bubble('assistant',event.assistant_text);refreshSessions();if(state.pendingSocketTurn?.id===event.client_turn_id){state.textTurn=false;state.pendingSocketTurn.resolve(event);state.pendingSocketTurn=null;}}else if(event.type==='state'){const labels={idle:'Ready when you are',listening:'Listening…',transcribing:'Transcribing…',thinking:'Thinking…',speaking:'Ready for another message',interrupted:'Response interrupted',recovering:'Ready to try again'};status(labels[event.state]||event.state,event.state==='thinking'?'busy':'');if(event.state==='interrupted'&&state.pendingSocketTurn){const pending=state.pendingSocketTurn;state.pendingSocketTurn=null;state.textTurn=false;pending.reject(Object.assign(new Error('Response interrupted'),{name:'AbortError'}));}if(['idle','speaking','interrupted','recovering'].includes(event.state)&&!state.voiceQueueRunning){state.busy=false;state.textTurn=false;if(state.voiceQueue.length)processVoiceQueue();}}else if(event.type==='error'){status(event.message,'error');if(state.pendingSocketTurn){const pending=state.pendingSocketTurn;state.pendingSocketTurn=null;state.textTurn=false;pending.reject(new Error(event.message));}if(!state.voiceQueueRunning){state.busy=false;state.textTurn=false;if(state.voiceQueue.length)processVoiceQueue();}}};
 socket.onerror=()=>{if(state.socket===socket&&!state.busy)status('Live connection unavailable; text fallback is ready','error');};
 socket.onclose=()=>{if(state.socket===socket){if(state.pendingSocketTurn){const pending=state.pendingSocketTurn;state.pendingSocketTurn=null;pending.reject(new Error('Live connection ended; try sending again.'));}if(state.busy){state.busy=false;state.textTurn=false;status('Live connection ended; try sending again.','error');if(state.voiceQueue.length)processVoiceQueue();}}};
}
function requestSocketTurn(text,turnId){
 const socket=state.socket;
 if(!socket||socket.readyState!==WebSocket.OPEN)return Promise.reject(new Error('Live connection is unavailable. Reconnect before sending a voice turn.'));
 return new Promise((resolve,reject)=>{state.pendingSocketTurn={id:turnId,resolve,reject};state.textTurn=true;socket.send(JSON.stringify({type:'user_text',text,client_turn_id:turnId}));});
}
async function newSession() { if(state.busy||state.stream||state.startingMic){status('Stop voice practice and finish the current turn before starting another conversation.','error');return;}const data=await api('/api/v1/sessions',{method:'POST'});await openSession(data.id);status('Ready when you are'); }
async function sendText(text) { text=text.trim();if(!text||state.busy)return;state.busy=true;const client_turn_id=crypto.randomUUID();$('#message-input').value='';status('Thinking…','busy');if(state.socket&&state.socket.readyState===WebSocket.OPEN){state.textTurn=true;state.socket.send(JSON.stringify({type:'user_text',text,client_turn_id}));return;}bubble('user',text);try{const answer=await api(`/api/v1/sessions/${state.session}/turns`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text,client_turn_id})});bubble('assistant',answer.assistant_text);status('Ready when you are');await refreshSessions();}catch(e){status(e.message,'error');}finally{state.busy=false;state.textTurn=false;if(state.voiceQueue.length)processVoiceQueue();} }
$('#composer').addEventListener('submit',e=>{e.preventDefault();sendText($('#message-input').value);});
$('#message-input').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();$('#composer').requestSubmit();}});
$('#message-input').addEventListener('input',e=>{e.target.style.height='auto';e.target.style.height=Math.min(e.target.scrollHeight,120)+'px';});
$('#new-session').onclick=newSession; $('#menu-toggle').onclick=()=>{const narrow=matchMedia('(max-width:700px)').matches;const sidebar=$('.sidebar');if(narrow){sidebar.classList.toggle('open');$('#menu-toggle').setAttribute('aria-expanded',String(sidebar.classList.contains('open')));}else{$('.shell').classList.toggle('sidebar-collapsed');$('#menu-toggle').setAttribute('aria-expanded',String(!$('.shell').classList.contains('sidebar-collapsed')));}};
document.querySelectorAll('.suggestion').forEach(b=>b.onclick=()=>sendText(b.textContent));
$('#diagnostics-open').onclick=async()=>{ $('#diagnostics').showModal();$('#diagnostic-data').textContent='Checking local services…';try{$('#diagnostic-data').textContent=JSON.stringify(await api('/api/v1/diagnostics'),null,2);}catch(e){$('#diagnostic-data').textContent=e.message;}};
$('#close-dialog').onclick=()=>$('#diagnostics').close();
function loadAudioSettings(){
 $('#silence-duration').value=localStorage.getItem('tutor_silence_seconds')||'3';
 $('#voice-sensitivity').value=localStorage.getItem('tutor_vad_threshold_v5')||'0.03';
 $('#sensitivity-value').textContent=`${(1+Number($('#voice-sensitivity').value)*120).toFixed(2)}×`;
 $('#voice-output').checked=localStorage.getItem('tutor_voice_output')!=='0';
}
async function loadAudioDevices(){
 const note=$('#audio-device-note');
 try{
  if(!navigator.mediaDevices?.enumerateDevices)throw new Error('This browser does not expose audio-device selection.');
  const devices=await navigator.mediaDevices.enumerateDevices();
  for(const [kind,id,label] of [['audioinput','audio-input','System default microphone'],['audiooutput','audio-output','System default speaker']]){
   const select=$(`#${id}`),previous=localStorage.getItem(id==='audio-input'?'tutor_audio_input':'tutor_audio_output')||'';
   select.replaceChildren(new Option(label,''));
   devices.filter(device=>device.kind===kind).forEach((device,index)=>select.add(new Option(device.label||`${kind==='audioinput'?'Microphone':'Speaker'} ${index+1}`,device.deviceId)));
   select.value=[...select.options].some(option=>option.value===previous)?previous:'';
  }
  note.textContent=devices.some(device=>device.label)?'Device choices apply when the next voice session starts. Speaker selection requires browser support for setSinkId.':'Grant microphone permission to display device names.';
 }catch(e){note.textContent=e.message;}
}
loadAudioSettings();
$('#audio-settings-open').onclick=()=>{$('#audio-settings').showModal();loadAudioDevices();};
$('#close-audio-settings').onclick=()=>$('#audio-settings').close();
$('#silence-duration').onchange=e=>localStorage.setItem('tutor_silence_seconds',e.target.value);
$('#voice-sensitivity').oninput=e=>{$('#sensitivity-value').textContent=`${(1+Number(e.target.value)*120).toFixed(2)}×`;localStorage.setItem('tutor_vad_threshold_v5',e.target.value);};
$('#voice-output').onchange=e=>localStorage.setItem('tutor_voice_output',e.target.checked?'1':'0');
$('#audio-input').onchange=e=>localStorage.setItem('tutor_audio_input',e.target.value);
$('#audio-output').onchange=e=>localStorage.setItem('tutor_audio_output',e.target.value);
async function loadMemories(query=''){const box=$('#memory-items');box.replaceChildren();try{const data=await api(`/api/v1/memories${query?`?q=${encodeURIComponent(query)}`:''}`);if(!data.memories.length){box.innerHTML='<div class="memory-empty">No remembered conversations yet.</div>';return;}data.memories.forEach(item=>{const card=document.createElement('article');card.className='memory-card';const learnerLabel=document.createElement('div');learnerLabel.className='memory-label';learnerLabel.textContent='You said';const learner=document.createElement('p');learner.textContent=item.user_text;const tutorLabel=document.createElement('div');tutorLabel.className='memory-label';tutorLabel.textContent='Tutor replied';const tutor=document.createElement('p');tutor.textContent=item.assistant_text;const remove=document.createElement('button');remove.textContent='Forget this conversation';remove.onclick=async()=>{if(confirm('Remove this remembered conversation?')){await api(`/api/v1/memories/${item.id}`,{method:'DELETE'});await loadMemories($('#memory-search').value);}};card.append(learnerLabel,learner,tutorLabel,tutor,remove);box.append(card);});}catch(e){box.textContent=e.message;}}
$('#memory-open').onclick=()=>{$('#memory-dialog').showModal();loadMemories($('#memory-search').value);};$('#close-memory').onclick=()=>$('#memory-dialog').close();$('#memory-search').addEventListener('input',e=>loadMemories(e.target.value));

function encodeWav(samples, inputRate) { const rate=16000, ratio=inputRate/rate, count=Math.floor(samples.length/ratio), pcm=new Int16Array(count); for(let i=0;i<count;i++){const x=i*ratio, lo=Math.floor(x), hi=Math.min(lo+1,samples.length-1), v=samples[lo]+(samples[hi]-samples[lo])*(x-lo);pcm[i]=Math.max(-1,Math.min(1,v))*32767;}const buffer=new ArrayBuffer(44+pcm.length*2),view=new DataView(buffer);const write=(o,s)=>[...s].forEach((c,i)=>view.setUint8(o+i,c.charCodeAt(0)));write(0,'RIFF');view.setUint32(4,36+pcm.length*2,true);write(8,'WAVE');write(12,'fmt ');view.setUint32(16,16,true);view.setUint16(20,1,true);view.setUint16(22,1,true);view.setUint32(24,rate,true);view.setUint32(28,rate*2,true);view.setUint16(32,2,true);view.setUint16(34,16,true);write(36,'data');view.setUint32(40,pcm.length*2,true);new Int16Array(buffer,44).set(pcm);return buffer; }
async function startMic() {
 if(state.stream||state.startingMic)return;
 state.startingMic=true;const request=++state.micRequest;status('Starting microphone…','busy');
 try {
  const selectedInput=localStorage.getItem('tutor_audio_input')||'';
  const audioConstraints={channelCount:1,echoCancellation:true,noiseSuppression:true,autoGainControl:true};
  if(selectedInput)audioConstraints.deviceId={exact:selectedInput};
    let stream;
    try{stream=await navigator.mediaDevices.getUserMedia({audio:audioConstraints});}
    catch(error){
     if(!selectedInput||!['NotFoundError','OverconstrainedError'].includes(error.name))throw error;
     localStorage.removeItem('tutor_audio_input');
     delete audioConstraints.deviceId;
     status('Saved microphone unavailable; trying the system default…','busy');
     stream=await navigator.mediaDevices.getUserMedia({audio:audioConstraints});
    }
  if(request!==state.micRequest){stream.getTracks().forEach(track=>track.stop());return;}
  state.stream=stream;
  const context=new AudioContext();state.audioContext=context;await context.resume();
  if(context.state!=='running')throw new Error(`Audio capture is ${context.state}; click Start again to resume microphone access.`);
  if(request!==state.micRequest){stream.getTracks().forEach(track=>track.stop());context.close().catch(()=>{});return;}
  const source=context.createMediaStreamSource(stream),processor=context.createScriptProcessor(2048,1,1);
  state.stream=stream;state.audioContext=context;state.processor=processor;state.lastAudioFrameAt=performance.now();
  state.detector=new VoiceCapture.VoiceTurnDetector({
   threshold:Number(localStorage.getItem('tutor_vad_threshold_v5'))||0.03,
   silenceSeconds:Number(localStorage.getItem('tutor_silence_seconds'))||3,
   sampleRate:context.sampleRate,
   onSpeechStart:()=>{if(state.busy)state.pendingBargeIn=true;if(state.textTurn&&state.socket?.readyState===WebSocket.OPEN)state.socket.send(JSON.stringify({type:'cancel'}));if(state.playbackController||state.player||state.playbackQueue.length){interruptPlayback();status('Interrupted · capturing your reply…','busy');}else status('Speech detected · listening for the end of your turn','busy');},
   onLevel:level=>{state.lastAudioFrameAt=performance.now();const now=state.lastAudioFrameAt;if(now-state.lastLevelStatusAt<300)return;state.lastLevelStatusAt=now;const volume=level.rms.toFixed(3),floor=Number(level.noiseFloor||0).toFixed(3),cutoff=Number(level.threshold||0).toFixed(3);if(level.calibrating){status(`Calibrating microphone · input level ${volume}`,'busy');return;}if(level.active){const meter=`mic ${volume} · room ${floor} · quiet below ${cutoff}`;status(level.voiced?`Speech detected · ${meter}`:`Quiet · ${meter} · sending in ${Math.max(0,level.endpointSeconds-level.silenceSeconds).toFixed(1)} s`,level.voiced?'busy':'');}else status(`Microphone active · waiting for speech (level ${volume}, trigger ${cutoff})`);},
   onUtterance:(samples)=>{const wav=encodeWav(samples,context.sampleRate);submitVoice(wav);}
  });
  processor.onaudioprocess=e=>{e.outputBuffer.getChannelData(0).fill(0);if(!state.detector)return;const elapsed=e.inputBuffer.length/context.sampleRate;state.detector.push(e.inputBuffer.getChannelData(0),elapsed);};
  source.connect(processor);processor.connect(context.destination);
  stream.getAudioTracks().forEach(track=>track.addEventListener('ended',()=>{if(state.stream===stream){stopMic(false);status('Microphone disconnected. Reconnect it and start voice practice again.','error');}}));
  state.captureWatchdog=setInterval(()=>{if(state.stream&&performance.now()-state.lastAudioFrameAt>5000){stopMic(false);status('Microphone capture stopped unexpectedly. Check the selected input device and try again.','error');}},1500);
  state.startingMic=false;$('#mic').classList.add('active');$('#mic').textContent='■';$('#mic').setAttribute('aria-label','Stop voice practice and send the current utterance');status('Microphone active · waiting for speech');
 }catch(e){if(request!==state.micRequest)return;state.startingMic=false;stopMic(false);status(`Microphone unavailable: ${e.message}`,'error');}
}
function stopMic(finalize=true){
 if(state.startingMic&&!state.stream){state.micRequest++;state.startingMic=false;status('Microphone start cancelled');return;}
 const context=state.audioContext;
 const hadSpeech=Boolean(state.detector?.active);
 if(finalize&&hadSpeech)status('Finalizing your recording…','busy');
 const submitted=finalize&&state.detector?.finalize('manual-stop');
 if(submitted)status('Transcribing…','busy');
 state.detector=null;
 if(state.captureWatchdog){clearInterval(state.captureWatchdog);state.captureWatchdog=null;}
 if(state.processor){state.processor.onaudioprocess=null;state.processor.disconnect();state.processor=null;}
 if(state.stream){state.stream.getTracks().forEach(track=>track.stop());state.stream=null;}
 if(context){state.audioContext=null;context.close().catch(()=>{});}
 interruptPlayback();
 $('#mic').classList.remove('active');$('#mic').textContent='●';$('#mic').setAttribute('aria-label','Start voice practice');
 if(hadSpeech&&!submitted)status('That recording was too short to transcribe. Speak for at least a quarter second and try again.','error');
 else if(!state.busy)status('Voice practice stopped');
}
function releasePlaybackWaiters(){for(const resolve of state.playbackSpaceResolvers.splice(0))resolve();if(!state.player&&!state.playbackQueue.length)for(const resolve of state.playbackDrainResolvers.splice(0))resolve();}
function pumpPlayback(epoch,playbackId){
 if(state.player||!state.playbackQueue.length){releasePlaybackWaiters();return;}
 const item=state.playbackQueue.shift();for(const resolve of state.playbackSpaceResolvers.splice(0))resolve();
 if(epoch!==state.turnEpoch||playbackId!==state.playbackId){URL.revokeObjectURL(item.url);pumpPlayback(epoch,playbackId);return;}
 const audio=new Audio(item.url);state.player=audio;state.playerUrl=item.url;
 const finish=()=>{audio.onended=null;audio.onerror=null;if(state.player===audio){state.player=null;state.playerUrl=null;URL.revokeObjectURL(item.url);pumpPlayback(epoch,playbackId);}else URL.revokeObjectURL(item.url);};
 audio.onplay=()=>{if(epoch===state.turnEpoch&&playbackId===state.playbackId)status('Speaking…','busy');};audio.onended=finish;audio.onerror=finish;
 const output=localStorage.getItem('tutor_audio_output');
 Promise.resolve(output&&typeof audio.setSinkId==='function'?audio.setSinkId(output):undefined).then(()=>audio.play()).catch(error=>{finish();status(`Text reply ready; speech playback failed: ${error.message}`,'error');});
}
async function enqueuePlayback(audioBase64,epoch,playbackId){
 while(state.playbackQueue.length>=2&&epoch===state.turnEpoch&&playbackId===state.playbackId)await new Promise(resolve=>state.playbackSpaceResolvers.push(resolve));
 if(epoch!==state.turnEpoch||playbackId!==state.playbackId)return;
 const bytes=Uint8Array.from(atob(audioBase64),c=>c.charCodeAt(0)),url=URL.createObjectURL(new Blob([bytes],{type:'audio/wav'}));
 state.playbackQueue.push({url});pumpPlayback(epoch,playbackId);
}
function waitForPlaybackDrain(){if(!state.player&&!state.playbackQueue.length)return Promise.resolve();return new Promise(resolve=>state.playbackDrainResolvers.push(resolve));}
function interruptPlayback(){
 state.playbackId++;
 if(state.playbackController){state.playbackController.abort();state.playbackController=null;}
 const audio=state.player,url=state.playerUrl;state.player=null;state.playerUrl=null;
 if(audio){audio.onended=null;audio.onerror=null;audio.pause();if(url)URL.revokeObjectURL(url);}
 for(const item of state.playbackQueue)URL.revokeObjectURL(item.url);state.playbackQueue=[];releasePlaybackWaiters();
}
async function playSpeechStream(text,epoch){
 const controller=new AbortController(),playbackId=++state.playbackId;state.playbackController=controller;
 try{
  const response=await fetch('/api/v1/speech/stream',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text}),signal:controller.signal});
  if(!response.ok){const error=await response.json().catch(()=>({}));throw new Error(error.detail||`Speech request failed (${response.status})`);}
  if(!response.body)throw new Error('This browser cannot stream spoken responses.');
  const reader=response.body.getReader(),decoder=new TextDecoder();let pending='',received=0,complete=false;
  const consume=async line=>{if(!line.trim())return;const event=JSON.parse(line);if(event.type==='error')throw new Error(event.message);if(event.type==='audio'){received++;await enqueuePlayback(event.audio_base64,epoch,playbackId);}if(event.type==='done')complete=true;};
  while(true){const {value,done}=await reader.read();pending+=decoder.decode(value||new Uint8Array(),{stream:!done});const lines=pending.split('\n');pending=lines.pop()||'';for(const line of lines)await consume(line);if(done)break;}
  if(pending)await consume(pending);if(!complete)throw new Error('Speech stream ended before completion.');if(!received)throw new Error('Kokoro returned no speech audio.');await waitForPlaybackDrain();
 }finally{if(state.playbackController===controller)state.playbackController=null;}
}
function submitVoice(wav){state.voiceQueue.push({wav,session:state.session,epoch:state.turnEpoch});processVoiceQueue();}
async function processVoiceQueue(){
 if(state.voiceQueueRunning||state.busy&&!state.voiceQueueRunning||!state.voiceQueue.length)return;
 state.busy=true;state.voiceQueueRunning=true;
 let lastError=null;
 while(state.voiceQueue.length){
  const item=state.voiceQueue.shift(),turnId=crypto.randomUUID();
  if(item.epoch!==state.turnEpoch)continue;
  lastError=null;
  if(state.pendingBargeIn)state.pendingBargeIn=false;
  try{
   status('Transcribing…','busy');
   const transcript=await api('/api/v1/voice/transcription',{method:'POST',headers:{'Content-Type':'audio/wav'},body:item.wav});
   if(item.epoch!==state.turnEpoch)continue;
   status('Thinking…','busy');
   const turn=await requestSocketTurn(transcript.text,turnId);
   if(item.epoch!==state.turnEpoch)continue;
   await refreshSessions();
   const speechMayPlay=VoiceCapture.shouldSpeakReply({pendingBargeIn:state.pendingBargeIn,queuedTurns:state.voiceQueue.length,speechActive:Boolean(state.detector?.active)});
   if(localStorage.getItem('tutor_voice_output')!=='0'&&speechMayPlay){
    try{status('Preparing speech…','busy');await playSpeechStream(turn.speech_text,item.epoch);}
    catch(e){if(e.name!=='AbortError'){lastError=e.message;status(`Text reply ready; speech unavailable: ${e.message}`,'error');}}
   }else if(!speechMayPlay){status('Interrupted · listening…','busy');}
  }catch(e){
   if(e.name!=='AbortError'){lastError=e.message;if(item.epoch===state.turnEpoch)status(e.message,'error');}
   try{const saved=await api(`/api/v1/sessions/${item.session}`);if(item.epoch===state.turnEpoch)renderMessages(saved.messages);}catch{}
  }
 }
 state.busy=false;state.textTurn=false;state.voiceQueueRunning=false;
 if(state.stream&&!state.detector?.active){if(lastError)status(`Voice turn error: ${lastError}`,'error');else status('Listening · speak naturally');}
}
$('#mic').onclick=()=>{if(state.stream||state.startingMic)stopMic();else startMic();};
document.addEventListener('keydown',e=>{if(e.key==='Escape'&&(state.playbackController||state.player||state.playbackQueue.length)){interruptPlayback();status('Speech interrupted');}if(e.key==='Escape'&&state.busy&&state.socket&&state.socket.readyState===WebSocket.OPEN){state.socket.send(JSON.stringify({type:'cancel'}));status('Cancelling response…','busy');}});
(async()=>{try{const list=await api('/api/v1/sessions');if(list.sessions.length)await openSession(list.sessions[0].id);else await newSession();}catch(e){status(`App startup error: ${e.message}`,'error');}})();
