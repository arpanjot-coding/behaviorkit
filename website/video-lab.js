(()=>{
 'use strict';
 const $=id=>document.getElementById(id),video=$('lab-video'),svg=$('lab-overlay');
 const names={'rtmdet-tiny':'RTMDet-Tiny','d-fine-n':'D-FINE-N','rf-detr-nano':'RF-DETR-Nano'};
 const contracts={'rtmdet-tiny':'640×640 BGR: preserve aspect ratio, pad with 114, normalize channels, then apply class-wise NMS to decoded boxes. Person class: 0.','d-fine-n':'640×640 RGB scaled to 0–1, plus original target width/height. The deployed graph returns top-ranked labels, pixel boxes and scores. Person class: 0.','rf-detr-nano':'384×384 RGB with ImageNet normalization. Rank sigmoid scores across query/class pairs and convert normalized centre boxes to XYXY. Sparse COCO person class: 1.'};
 let report,selectedCase='bread',model='rtmdet-tiny',evidence=null,revision=0,lastRender='',selectedTrack=null,pendingSeek=null;
 const cache=new Map(),fmt=t=>Number(t).toFixed(1),put=(id,value)=>$(id).textContent=String(value);
 const el=(tag,text,className)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(className)e.className=className;return e;};
 const currentCase=()=>report?.cases.find(c=>c.id===selectedCase);
 const duration=()=>Number.isFinite(video.duration)&&video.duration>0?video.duration:(currentCase()?.prepared.seconds||0);
 const jumpValue=()=>{const n=$('jump-seconds').valueAsNumber;return Number.isFinite(n)&&n>=.1&&n<=60?n:null;};
 async function json(url){const response=await fetch(url);if(!response.ok)throw Error('HTTP '+response.status);return response.json();}
 function fail(error){put('lab-status','Evidence could not load: '+error.message+'. ');const retry=el('button','Retry','retry-evidence');retry.addEventListener('click',()=>report?loadEvidence():init());$('lab-status').append(retry);}
 function sampleIndex(t){let lo=0,hi=evidence.frames.length-1;while(lo<hi){const mid=Math.ceil((lo+hi)/2);if(evidence.frames[mid].timestamp<=t+1e-6)lo=mid;else hi=mid-1;}return lo;}
 function seek(t,{pause=true}={}){
  if(!Number.isFinite(t)||!report)return;
  const target=Math.max(0,Math.min(duration(),t));if(pause)video.pause();
  if(video.readyState===0)pendingSeek=target;else video.currentTime=target;
  lastRender='';render();
 }
 function jump(direction){const seconds=jumpValue();if(seconds!==null)seek(video.currentTime+seconds*direction,{pause:false});}
 function step(direction){
  if(!evidence)return;const i=sampleIndex(video.currentTime);
  const next=direction>0?i+1:(evidence.frames[i].timestamp<video.currentTime-1e-6?i:i-1);
  seek(evidence.frames[Math.max(0,Math.min(evidence.frames.length-1,next))].timestamp);
 }
 function togglePlay(){if(video.paused){if(video.ended)seek(0);video.play().catch(error=>put('lab-status','Playback could not start: '+error.message+'. Try Play again.'));}else video.pause();}
 function syncPlayback(){const playing=!video.paused&&!video.ended;put('lab-play',video.ended?'Replay':playing?'Pause':'Play');$('lab-play').setAttribute('aria-label',playing?'Pause clip':'Play clip');put('playback-state',video.seeking?'SEEKING':video.ended?'CLIP COMPLETE':playing?Number(video.playbackRate)+'× PLAYING':'PAUSED');}
 function controls(){
  const jumpBy=jumpValue(),ready=video.readyState>=1&&!!evidence,t=video.currentTime;
  $('jump-seconds').setAttribute('aria-invalid',String(jumpBy===null));
  put('jump-error',jumpBy===null?'Enter a jump between 0.1 and 60 seconds.':'');
  for(const [id,direction] of [['jump-back',-1],['jump-forward',1]]){
   put(id,(direction<0?'−':'+')+(jumpBy===null?'…':Number(jumpBy))+' s');
   $(id).setAttribute('aria-label','Jump '+(direction<0?'backward ':'forward ')+(jumpBy===null?'':jumpBy+' ')+'seconds');
   $(id).disabled=!ready||jumpBy===null||(direction<0?t<=.001:t>=duration()-.001);
  }
  for(const id of ['lab-play','frame-back','frame-next','clip-start','clip-end'])$(id).disabled=!ready;
  $('lab-seek').disabled=!ready;
 }
 function shape(tag,attrs,text){const e=document.createElementNS('http://www.w3.org/2000/svg',tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,String(v)));if(text!==undefined)e.textContent=text;svg.append(e);return e;}
 function draw(frame,stale){
  svg.replaceChildren();if(stale||$('overlay-mode').value==='off')return;
  const c=currentCase(),height=1000*c.prepared.height/c.prepared.width,fontSize=12*1000/$('lab-stage').clientWidth;
  svg.setAttribute('viewBox','0 0 1000 '+height);
  if($('show-zone').checked)shape('polygon',{points:'3,3 997,3 997,'+(height-3)+' 3,'+(height-3),class:'lab-zone'});
  const raw=$('overlay-mode').value==='raw',boxes=raw?frame.raw:frame.tracks;
  boxes.forEach(d=>{
   const [x1,y1,x2,y2]=[d.box[0]*1000,d.box[1]*height,d.box[2]*1000,d.box[3]*height],chosen=!raw&&selectedTrack===d.track_id,dim=!raw&&selectedTrack!==null&&!chosen;
   const label=(raw?'person':'#'+d.track_id)+' · '+d.confidence.toFixed(2),color=raw?'#ffc16b':chosen?'#acffe2':'#86aaff';
   shape('rect',{x:x1,y:y1,width:x2-x1,height:y2-y1,fill:'none',stroke:color,'stroke-width':chosen?3:2,opacity:dim?.35:1});
   shape('text',{x:Math.max(6,Math.min(990-label.length*fontSize*.63,x1+4)),y:Math.max(fontSize+5,y1+fontSize+5),style:'font-size:'+fontSize+'px',class:raw?'detector-label':'',opacity:dim?.5:1},label);
   if(!raw&&$('show-zone').checked)shape('circle',{cx:(x1+x2)/2,cy:y2,r:chosen?7:5});
  });
 }
 function inspectTrack(frame,stale){
  const list=$('track-chips');list.replaceChildren();
  if(frame.tracks.length){const all=el('button','Auto','track-chip'+(selectedTrack===null?' active':''));all.setAttribute('aria-pressed',String(selectedTrack===null));all.addEventListener('click',()=>{video.pause();selectedTrack=null;lastRender='';render();});list.append(all);}
  frame.tracks.forEach(d=>{const b=el('button','#'+d.track_id,'track-chip'+(selectedTrack===d.track_id?' active':''));b.dataset.track=d.track_id;b.setAttribute('aria-pressed',String(selectedTrack===d.track_id));b.addEventListener('click',()=>{video.pause();selectedTrack=d.track_id;lastRender='';render();});list.append(b);});
  const longest=[...frame.dwell].sort((a,b)=>b.duration-a.duration)[0];
  const chosen=selectedTrack??longest?.track_id??frame.tracks[0]?.track_id;
  const track=frame.tracks.find(d=>d.track_id===chosen),state=frame.dwell.find(d=>d.track_id===chosen),span=state?.duration||0;
  const meter=$('dwell-progress').parentElement;meter.setAttribute('aria-valuenow',String(stale?0:Math.min(3,span)));meter.setAttribute('aria-valuetext',stale?'Waiting for synchronized evidence':fmt(span)+' of 3 seconds');$('dwell-progress').style.width=(stale?0:Math.min(100,span/3*100))+'%';
  const checks=$('decision-checks');checks.replaceChildren();
  if(stale){put('track-heading','Waiting for the video to reach this sample.');put('track-episode','');return;}
  if(!track){put('track-heading',chosen?'Track #'+chosen+' is absent at this sample.':'No confirmed track at this sample.');put('track-episode','No current track evidence means no supported dwell span. It does not establish that a person physically left.');return;}
  put('track-heading',(selectedTrack===null?'Auto-selected ':'Selected ')+'track #'+track.track_id+' · confidence '+track.confidence.toFixed(2));
  put('track-episode',state?'Episode: '+fmt(state.start)+' → '+fmt(state.last)+' s. '+(state.emitted?'A dwell event has already been emitted.':'Needs '+fmt(Math.max(0,3-span))+' more supported seconds.'):'No eligible dwell state for this track.');
  const anchor=[(track.box[0]+track.box[2])/2,track.box[3]];
  const conditions=[['Person class',track.label==='person',track.label],['Confidence ≥ 0.30',track.confidence>=.3,track.confidence.toFixed(2)],['Confirmed by tracker',true,'two-hit baseline'],['Anchor in count zone',frame.count.track_ids.includes(track.track_id),anchor.map(n=>n.toFixed(3)).join(', ')],['Supported span ≥ 3.0 s',span+1e-9>=3,fmt(span)+' s']];
  conditions.forEach(([label,pass,value])=>{const row=el('div',undefined,'decision-check'+(pass?' passed':' waiting'));row.append(el('span',pass?'✓':'○'),el('span',label),el('strong',value));checks.append(row);});
 }
 function render(){
  if(!report)return;const t=video.currentTime||0,total=duration();put('lab-clock',fmt(t)+' / '+fmt(total)+' s');$('lab-seek').max=total;$('lab-seek').value=t;controls();syncPlayback();
  if(!evidence)return;
  const index=sampleIndex(t),frame=evidence.frames[index],age=Math.max(0,t-frame.timestamp),stale=age>.25||video.seeking;
  $('lab-stage').classList.toggle('is-seeking',video.seeking);
  put('sample-age','Video '+fmt(t)+' s · sample '+fmt(frame.timestamp)+' s'+(video.seeking?' · seeking':stale?' · stale':' · age '+fmt(age)+' s'));
  const key=[selectedCase,model,index,stale,$('overlay-mode').value,$('show-zone').checked,selectedTrack].join(':');if(lastRender===key)return;lastRender=key;
  put('frame-stamp','sample '+(index+1)+' / '+evidence.frames.length+' · t = '+fmt(frame.timestamp)+' s');
  draw(frame,stale);put('raw-count',stale?'…':frame.raw.length);put('track-count',stale?'…':frame.tracks.length);put('rule-count',stale?'unknown':frame.count.count);
  const longest=[...frame.dwell].sort((a,b)=>b.duration-a.duration)[0];put('rule-span',stale?'…':longest?fmt(longest.duration)+' s':'0.0 s');
  let reason;
  if(stale)reason='Waiting for the video frame or a recent inference sample. Overlays are hidden until they can be synchronized.';
  else if(!longest)reason=frame.raw.length?'Boxes were detected, but no eligible confirmed track is present. New candidates need a second match; there is no supported dwell span yet.':'No eligible person was detected. The rule reports zero and resets unsupported timers. A missed detection can still hide a real person.';
  else if(frame.events.length)reason='Track #'+frame.events[0].track_id+' reached '+fmt(frame.events[0].duration_seconds)+' supported seconds. The engine emitted a dwell event at '+fmt(frame.timestamp)+' s, once for this episode.';
  else if(longest.emitted)reason='Track #'+longest.track_id+' has a '+fmt(longest.duration)+' s supported span. Its dwell event has already fired; it will not repeat during this episode.';
  else reason='Track #'+longest.track_id+' has supported observations from '+fmt(longest.start)+' to '+fmt(frame.timestamp)+' s ('+fmt(longest.duration)+' / 3.0 s). Missing tracks or gaps over 0.5 s reset its timer.';
  put('rule-reason',reason);put('rule-reset',frame.resets.length?'Reset at this sample: #'+frame.resets.join(', #')+'. Supporting track evidence was lost. This is not proof of a physical exit.':'');
  const notes=evidence.human_annotations.human_notes.filter(n=>t>=n.start&&t<=n.end);put('human-note',notes.length?notes.map(n=>n.text).join(' '):'No time-specific human note. '+currentCase().challenge);
  put('sample-json',JSON.stringify(frame,null,2));inspectTrack(frame,stale);
  document.querySelectorAll('.event-item').forEach(b=>b.classList.toggle('future',Number(b.dataset.time)>frame.timestamp+1e-6));
 }
 function timeline(){
  for(const id of ['event-markers','reset-markers','human-markers'])$(id).replaceChildren();
  const events=new Map();evidence.events.forEach(event=>{const values=events.get(event.timestamp)||[];values.push(event.track_id);events.set(event.timestamp,values);});
  const marker=(id,t,label,className)=>{const b=el('button',undefined,'timeline-marker '+className);b.style.left='clamp(6px,'+(t/duration()*100)+'%,calc(100% - 6px))';b.title=label;b.setAttribute('aria-label',label);if(className==='reset-tick')b.tabIndex=-1;b.dataset.time=t;b.addEventListener('click',()=>seek(t));$(id).append(b);};
  for(const [t,ids] of events)marker('event-markers',t,fmt(t)+' s: dwell event for #'+ids.join(', #'),'event-tick');
  evidence.frames.filter(f=>f.resets.length).forEach(f=>marker('reset-markers',f.timestamp,fmt(f.timestamp)+' s: '+f.resets.length+' track timer reset'+(f.resets.length===1?'':'s'),'reset-tick'));
  evidence.human_annotations.human_notes.forEach(note=>{const b=el('button',undefined,'human-interval');b.style.left=(note.start/duration()*100)+'%';b.style.width=((Math.min(duration(),note.end)-note.start)/duration()*100)+'%';b.title=fmt(note.start)+'–'+fmt(note.end)+' s: '+note.text;b.setAttribute('aria-label',b.title);b.addEventListener('click',()=>seek(note.start));$('human-markers').append(b);});
  if(!events.size)$('event-markers').append(el('span','No events','lane-empty'));
  if(!evidence.frames.some(f=>f.resets.length))$('reset-markers').append(el('span','No resets','lane-empty'));
  const guide=$('guide-points');guide.replaceChildren();
  evidence.human_annotations.human_notes.forEach(note=>{const b=el('button',undefined,'guide-point');b.append(el('strong',fmt(note.start)+'–'+fmt(note.end)+' s'),el('span',note.text));b.addEventListener('click',()=>seek(note.start));guide.append(b);});
 }
 function comparison(){
  const cards=$('model-cards');cards.replaceChildren();
  report.runs.filter(r=>r.case===selectedCase).forEach(run=>{
   const b=el('button',undefined,'model-card'+(run.model===model?' active':''));b.dataset.model=run.model;b.setAttribute('aria-pressed',String(run.model===model));
   b.append(el('span',run.model===model?'INSPECTING THIS MODEL':'INSPECT MODEL','model-kicker'),el('h3',names[run.model]));
   const stats=el('div',undefined,'comparison-stats');
   for(const [value,label] of [[run.spot_checks.scored?run.spot_checks.exact+'/'+run.spot_checks.scored:'N/A','exact snapshots'],[run.events,'dwell events'],[run.track_ids_seen,'temporary IDs'],[run.reset_records,'reset records']]){const item=el('div');item.append(el('strong',value),el('span',label));stats.append(item);}
   b.append(stats);b.addEventListener('click',()=>{changeModel(run.model);$('lab-stage').scrollIntoView({block:'start'});});cards.append(b);
  });
 }
 function fillEvents(){
  const list=$('event-list');list.replaceChildren();
  if(!evidence.events.length)list.append(el('p','No dwell event was emitted. Inspect the tracks and resets: the engine never supported a continuous three-second episode.','event-empty'));
  evidence.events.forEach(event=>{const b=el('button',undefined,'event-item');b.dataset.time=event.timestamp;b.append(el('strong',fmt(event.timestamp)+' s · person_dwell · track #'+event.track_id),el('span',fmt(event.duration_seconds)+' s supported; first eligible sample at '+fmt(event.start)+' s'));b.addEventListener('click',()=>seek(event.timestamp));list.append(b);});
 }
 function fillChecks(){
  const score=evidence.summary.spot_checks,list=$('spot-checks');list.replaceChildren();
  put('score-summary',score.scored?score.exact+' / '+score.scored+' exact counts at the manually checked snapshots. Mean absolute count error: '+score.mean_absolute_error.toFixed(2)+'. These count checks do not score dwell correctness or theft recognition.':'No exact count ground truth was assigned to this crowded clip. Its outputs remain inspectable without an accuracy score.');
  score.checks.forEach(check=>{const label=check.status==='scored'?'human '+check.count+' · model '+check.observed:check.status==='human_abstained'?'human abstained · unscored':'no nearby sample · unscored';const b=el('button',undefined,'spot-check '+(check.status==='scored'?(check.absolute_error?'miss':'match'):'abstain'));b.append(el('span',fmt(check.t)+' s'),el('span',label));if(check.reason)b.title=check.reason;b.addEventListener('click',()=>seek(check.t));list.append(b);});
 }
 function clearEvidence(){
  evidence=null;selectedTrack=null;lastRender='';svg.replaceChildren();
  for(const id of ['raw-count','track-count','rule-count','rule-span'])put(id,'…');
  for(const id of ['event-list','spot-checks','event-markers','reset-markers','human-markers','guide-points','track-chips','decision-checks'])$(id).replaceChildren();
  for(const id of ['sample-json','model-json','config-json','track-episode','rule-reset','sample-age'])put(id,'');
  put('rule-reason','Loading this model’s saved observations.');put('track-heading','Loading track evidence…');put('score-summary','Loading snapshot checks…');$('dwell-progress').style.width='0%';$('dwell-progress').parentElement.setAttribute('aria-valuenow','0');$('dwell-progress').parentElement.setAttribute('aria-valuetext','Waiting for track evidence');
  for(const id of ['download-frames','download-observations','download-events','download-counts','download-config']){$(id).removeAttribute('href');$(id).setAttribute('aria-disabled','true');}
  controls();
 }
 async function loadEvidence(){
  const token=++revision,id=selectedCase,currentModel=model;clearEvidence();put('lab-status','Loading saved '+names[model]+' observations…');comparison();
  try{
   const key=id+'/'+currentModel;if(!cache.has(key))cache.set(key,json('/lab/data/'+key+'/frames.json').catch(error=>{cache.delete(key);throw error;}));
   const data=await cache.get(key);if(token!==revision)return;evidence=data;
   const base='/lab/data/'+key+'/';
   for(const [control,file] of [['download-frames','frames.json'],['download-observations','observations.jsonl'],['download-events','events.jsonl'],['download-counts','counts.jsonl'],['download-config','config.json']]){$(control).href=base+file;$(control).removeAttribute('aria-disabled');}
   put('model-contract',contracts[currentModel]);put('config-json',JSON.stringify(data.config,null,2));
   put('model-json',JSON.stringify({model:currentModel,provider:data.summary.provider,model_sha256:data.summary.model_sha256,observations_sha256:data.summary.observations_sha256,profile:report.verification[currentModel],environment:report.environment,stage_timing_note:'Build-host measurements; not browser inference or a controlled model ranking.'},null,2));
   put('reproduce-command','behaviorkit run --video examples/video-lab/media/'+id+'.mp4 --model '+currentModel+' --models-dir models --config examples/video-lab/config.json --provider cuda --threads 4 --sample-fps 5 --source '+id+' --output runs/'+id+'/'+currentModel);
   fillEvents();fillChecks();timeline();put('lab-status',data.summary.sampled_frames+' sampled frames · replay verified · '+data.summary.events+' dwell event'+(data.summary.events===1?'':'s')+' · action classifier: absent');render();
  }catch(error){if(token===revision)fail(error);}
 }
 function changeModel(next){model=next;$('lab-model').value=model;loadEvidence();}
 async function selectCase(id){
  selectedCase=id;pendingSeek=null;const c=currentCase(),source=report.sources[c.source];video.pause();
  document.querySelectorAll('.case-choice').forEach(b=>{const active=b.dataset.case===id;b.classList.toggle('active',active);b.setAttribute('aria-pressed',String(active));});
  put('case-tag',c.tag);put('case-title',c.title);put('case-description',c.description);$('lab-stage').style.aspectRatio=c.prepared.width+'/'+c.prepared.height;
  video.poster='/lab/media/'+id+'.jpg';video.src='/lab/media/'+id+'.mp4';video.load();
  put('source-title',source.title);put('source-credit',source.creator+' · '+source.license+' · excerpt '+fmt(c.start)+'–'+fmt(c.start+c.seconds)+' s of the source.');put('source-note',source.use_note);$('source-link').href=source.page;$('source-license').href=source.license_url;
  await loadEvidence();
 }
 $('lab-play').addEventListener('click',togglePlay);
 $('explore-scene03').addEventListener('click',async event=>{event.preventDefault();if(!report)return;await selectCase('bread');$('lab-workbench').scrollIntoView({block:'start'});});
 $('jump-back').addEventListener('click',()=>jump(-1));$('jump-forward').addEventListener('click',()=>jump(1));
 $('jump-seconds').addEventListener('input',controls);$('jump-seconds').addEventListener('change',controls);
 $('playback-speed').addEventListener('change',()=>{video.playbackRate=Number($('playback-speed').value);syncPlayback();});
 $('loop-video').addEventListener('change',()=>video.loop=$('loop-video').checked);
 $('clip-start').addEventListener('click',()=>seek(0));$('clip-end').addEventListener('click',()=>seek(duration()));
 $('frame-back').addEventListener('click',()=>step(-1));$('frame-next').addEventListener('click',()=>step(1));
 $('lab-seek').addEventListener('input',event=>seek(Number(event.target.value)));
 $('lab-model').addEventListener('change',event=>changeModel(event.target.value));
 for(const id of ['overlay-mode','show-zone'])$(id).addEventListener('change',()=>{lastRender='';render();});
 $('lab-stage').addEventListener('click',()=>$('lab-stage').focus({preventScroll:true}));
 $('lab-stage').addEventListener('keydown',event=>{
  if(event.altKey||event.ctrlKey||event.metaKey)return;
  if([' ','ArrowLeft','ArrowRight','Home','End'].includes(event.key))event.preventDefault();else return;
  if(event.key===' ')togglePlay();else if(event.key==='Home')seek(0);else if(event.key==='End')seek(duration());else if(event.shiftKey)step(event.key==='ArrowLeft'?-1:1);else jump(event.key==='ArrowLeft'?-1:1);
 });
 for(const event of ['play','pause','ended','ratechange'])video.addEventListener(event,syncPlayback);
 for(const event of ['timeupdate','seeking','seeked'])video.addEventListener(event,render);
 video.addEventListener('loadedmetadata',()=>{video.playbackRate=Number($('playback-speed').value);video.loop=$('loop-video').checked;if(pendingSeek!==null){const target=pendingSeek;pendingSeek=null;seek(target,{pause:false});}if(evidence)timeline();render();});
 video.addEventListener('error',()=>put('lab-status','Video could not load. Frame evidence remains downloadable.'));
 window.addEventListener('resize',()=>{lastRender='';render();});
 async function init(){
  try{report=await json('/lab/report.json');report.cases.forEach((c,index)=>{const b=el('button',undefined,'case-choice'),image=el('img');image.src='/lab/media/'+c.id+'.jpg';image.alt='';image.loading='lazy';b.dataset.case=c.id;b.append(image,el('small','0'+(index+1)+' / '+c.tag),el('strong',c.title),el('span',fmt(c.prepared.seconds)+' s · '+(c.theft_reference?'fictional scene':'normal activity'),'case-meta'));b.addEventListener('click',()=>selectCase(c.id));$('case-picker').append(b);});await selectCase(selectedCase);}catch(error){fail(error);}
 }
 function animate(){if(!video.paused)render();requestAnimationFrame(animate);}controls();animate();init();
})();
