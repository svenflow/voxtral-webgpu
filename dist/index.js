var J={backbone:{dim:3072,n_layers:26,head_dim:128,hidden_dim:9216,n_heads:32,n_kv_heads:8,vocab_size:131072,rope_theta:1e6,norm_eps:1e-5},fm:{input_dim:3072,dim:3072,n_layers:3,head_dim:128,hidden_dim:9216,n_heads:32,n_kv_heads:8,nfe:8,cfg_alpha:1.2,rope_theta:1e4,sigma:1e-5,sigma_max:1,n_acoustic_out:36,semantic_vocab:8320},codec:{dim:1024,hidden_dim:4096,head_dim:128,n_heads:8,n_kv_heads:8,semantic_codebook_size:8192,semantic_dim:256,n_acoustic_codebook:36,acoustic_codebook_size:21,sampling_rate:24e3,frame_rate:12.5,patch_size:240,decoder_stages:4,decoder_layers_per_stage:2,decoder_conv_strides:[1,2,2,2],decoder_conv_kernels:[3,4,4,4],attn_sliding_window:16,norm_eps:.01,qk_norm_eps:1e-6,qk_norm:!0,layer_scale:!0,weight_norm_conv:!0}};var we=`
struct Params {
  M: u32,
  K: u32,
}

@group(0) @binding(0) var<storage, read> matrix: array<f32>;
@group(0) @binding(1) var<storage, read> vector: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const TILE_SIZE: u32 = 256u;
var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let row = wid.x;
  if (row >= params.M) { return; }
  let tid = lid.x;
  var sum: f32 = 0.0;
  let num_tiles = (params.K + TILE_SIZE - 1u) / TILE_SIZE;
  for (var t: u32 = 0u; t < num_tiles; t++) {
    let k = t * TILE_SIZE + tid;
    if (k < params.K) {
      sum += matrix[row * params.K + k] * vector[k];
    }
  }
  sdata[tid] = sum;
  workgroupBarrier();
  for (var stride: u32 = 128u; stride > 0u; stride >>= 1u) {
    if (tid < stride) { sdata[tid] += sdata[tid + stride]; }
    workgroupBarrier();
  }
  if (tid == 0u) { output[row] = sdata[0]; }
}
`;var ya=5,xa=50;function Ua(){let c=J;return[{name:"backbone.qkv_proj",M:c.backbone.n_heads*c.backbone.head_dim+2*c.backbone.n_kv_heads*c.backbone.head_dim,K:c.backbone.dim,N:1,count:c.backbone.n_layers,component:"backbone"},{name:"backbone.o_proj",M:c.backbone.dim,K:c.backbone.n_heads*c.backbone.head_dim,N:1,count:c.backbone.n_layers,component:"backbone"},{name:"backbone.ffn_gate_up",M:c.backbone.hidden_dim*2,K:c.backbone.dim,N:1,count:c.backbone.n_layers,component:"backbone"},{name:"backbone.ffn_down",M:c.backbone.dim,K:c.backbone.hidden_dim,N:1,count:c.backbone.n_layers,component:"backbone"},{name:"backbone.lm_head",M:c.backbone.vocab_size,K:c.backbone.dim,N:1,count:1,component:"backbone"},{name:"fm.qkv_proj",M:c.fm.n_heads*c.fm.head_dim+2*c.fm.n_kv_heads*c.fm.head_dim,K:c.fm.dim,N:1,count:c.fm.n_layers*c.fm.nfe*2,component:"fm"},{name:"fm.o_proj",M:c.fm.dim,K:c.fm.n_heads*c.fm.head_dim,N:1,count:c.fm.n_layers*c.fm.nfe*2,component:"fm"},{name:"fm.ffn_gate_up",M:c.fm.hidden_dim*2,K:c.fm.dim,N:1,count:c.fm.n_layers*c.fm.nfe*2,component:"fm"},{name:"fm.ffn_down",M:c.fm.dim,K:c.fm.hidden_dim,N:1,count:c.fm.n_layers*c.fm.nfe*2,component:"fm"},{name:"codec.qkv_proj",M:c.codec.n_heads*c.codec.head_dim+2*c.codec.n_kv_heads*c.codec.head_dim,K:c.codec.dim,N:1,count:c.codec.decoder_stages*c.codec.decoder_layers_per_stage,component:"codec"},{name:"codec.ffn_gate_up",M:c.codec.hidden_dim*2,K:c.codec.dim,N:1,count:c.codec.decoder_stages*c.codec.decoder_layers_per_stage,component:"codec"},{name:"codec.ffn_down",M:c.codec.dim,K:c.codec.hidden_dim,N:1,count:c.codec.decoder_stages*c.codec.decoder_layers_per_stage,component:"codec"}]}async function Ba(){if(!navigator.gpu)throw new Error("WebGPU not supported in this browser");let c=await navigator.gpu.requestAdapter({powerPreference:"high-performance"});if(!c)throw new Error("No WebGPU adapter found");let r=c.features.has("shader-f16"),o=c.features.has("timestamp-query"),t=[];return r&&t.push("shader-f16"),o&&t.push("timestamp-query"),await c.requestDevice({requiredFeatures:t,requiredLimits:{maxBufferSize:1024*1024*1024,maxStorageBufferBindingSize:512*1024*1024}})}function Ca(c,r){let o=c.createShaderModule({code:r});return c.createComputePipeline({layout:"auto",compute:{module:o,entryPoint:"main"}})}async function qa(c,r,o,t,e){let n=e?2:4,s=o*t*n;if(s>c.limits.maxStorageBufferBindingSize)return[-1];let i=c.createBuffer({size:s,usage:GPUBufferUsage.STORAGE,mappedAtCreation:!0});if(e){let u=new Uint16Array(i.getMappedRange());for(let d=0;d<u.length;d++)u[d]=Pe((Math.random()-.5)*.1)}else{let u=new Float32Array(i.getMappedRange());for(let d=0;d<u.length;d++)u[d]=(Math.random()-.5)*.1}i.unmap();let a=c.createBuffer({size:t*n,usage:GPUBufferUsage.STORAGE,mappedAtCreation:!0});if(e){let u=new Uint16Array(a.getMappedRange());for(let d=0;d<u.length;d++)u[d]=Pe((Math.random()-.5)*.1)}else{let u=new Float32Array(a.getMappedRange());for(let d=0;d<u.length;d++)u[d]=(Math.random()-.5)*.1}a.unmap();let _=c.createBuffer({size:o*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),m=c.createBuffer({size:8,usage:GPUBufferUsage.UNIFORM,mappedAtCreation:!0});new Uint32Array(m.getMappedRange()).set([o,t]),m.unmap();let P=c.createBindGroup({layout:r.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:i}},{binding:1,resource:{buffer:a}},{binding:2,resource:{buffer:_}},{binding:3,resource:{buffer:m}}]}),k=[];for(let u=0;u<ya;u++){let d=c.createCommandEncoder(),p=d.beginComputePass();p.setPipeline(r),p.setBindGroup(0,P),p.dispatchWorkgroups(o),p.end(),c.queue.submit([d.finish()])}await c.queue.onSubmittedWorkDone();for(let u=0;u<xa;u++){let d=c.createCommandEncoder(),p=d.beginComputePass();p.setPipeline(r),p.setBindGroup(0,P),p.dispatchWorkgroups(o),p.end();let w=performance.now();c.queue.submit([d.finish()]),await c.queue.onSubmittedWorkDone();let G=performance.now();k.push(G-w)}return i.destroy(),a.destroy(),_.destroy(),m.destroy(),k}function Pe(c){let r=new ArrayBuffer(4);new Float32Array(r)[0]=c;let o=new Uint32Array(r)[0],t=o>>16&32768,e=(o>>23&255)-127+15,n=o>>13&1023;return e<=0?t:e>=31?t|31744:t|e<<10|n}function Ga(c){let r=[...c].sort((t,e)=>t-e),o=Math.floor(r.length/2);return r.length%2?r[o]:(r[o-1]+r[o])/2}async function Aa(c){let r=c||console.log;r("Initializing WebGPU...");let o=await Ba(),t=o.adapterInfo,e=t?`${t.vendor} ${t.architecture} ${t.device}`:"Unknown GPU",n=o.features.has("shader-f16"),s=o.features.has("timestamp-query");r(`GPU: ${e}`),r(`F16 support: ${n}`),r(`Timestamp queries: ${s}`);let i=Ca(o,we),a=Ua(),_=[];for(let h of a){r(`
Benchmarking ${h.name} [${h.M} \xD7 ${h.K}] \xD7 ${h.count}...`);let f=await qa(o,i,h.M,h.K,!1);if(f[0]===-1){r(`  SKIPPED \u2014 buffer too large (${(h.M*h.K*4/1e9).toFixed(2)} GB)`),_.push({name:h.name,M:h.M,K:h.K,avgMs:-1,medianMs:-1,minMs:-1,count:h.count,totalMs:-1,component:h.component,gflops:0});continue}let l=f.reduce((x,q)=>x+q,0)/f.length,v=Ga(f),y=Math.min(...f),C=2*h.M*h.K/(v/1e3)/1e9,U={name:h.name,M:h.M,K:h.K,avgMs:l,medianMs:v,minMs:y,count:h.count,totalMs:v*h.count,component:h.component,gflops:C};_.push(U),r(`  median: ${v.toFixed(3)}ms | total (\xD7${h.count}): ${U.totalMs.toFixed(2)}ms | ${C.toFixed(1)} GFLOPS`)}let m=_.filter(h=>h.component==="backbone").reduce((h,f)=>h+f.totalMs,0),P=_.filter(h=>h.component==="fm").reduce((h,f)=>h+f.totalMs,0),k=_.filter(h=>h.component==="codec").reduce((h,f)=>h+f.totalMs,0),u=m+P+k,d=80,p=u<d,w=d/u,G={backbone_total_ms:m,fm_total_ms:P,codec_total_ms:k,total_per_frame_ms:u,target_ms:d,feasible:p,realtime_factor:w};return r(`
========================================`),r("VOXTRAL TTS \u2014 PHASE 0a GO/NO-GO RESULTS"),r("========================================"),r(`Backbone (26 layers):  ${m.toFixed(2)}ms`),r(`FM (3 layers \xD7 16):   ${P.toFixed(2)}ms`),r(`Codec (8 layers):     ${k.toFixed(2)}ms`),r("\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"),r(`Total per frame:      ${u.toFixed(2)}ms`),r(`Target (12.5 fps):    ${d}ms`),r(`Realtime factor:      ${w.toFixed(2)}x`),r(`
VERDICT: ${p?"\u2705 GO \u2014 real-time TTS is feasible!":"\u274C NO-GO \u2014 too slow for real-time"}`),p||(r(`
Note: These are matmul-only times. Real inference adds ~30-50% overhead`),r("for norms, activations, attention, sampling, etc."),r(`
For feasibility, we need matmul total < ~55ms (leaving ~25ms for overhead).`)),o.destroy(),{device:e,hasF16:n,hasTimestamp:s,results:_,summary:G}}async function ye(c){let o=await(await fetch(c,{headers:{Range:"bytes=0-7"}})).arrayBuffer(),t=Number(new DataView(o).getBigUint64(0,!0)),n=await(await fetch(c,{headers:{Range:`bytes=8-${8+t-1}`}})).text();return{header:JSON.parse(n),dataOffset:8+t}}function fe(c){let r=new Uint16Array(c),o=new Uint16Array(r.length);for(let t=0;t<r.length;t++){let e=r[t],n=e>>15&1,s=e>>7&255,i=e&127;if(s===255)o[t]=n<<15|31744|(i?512:0);else if(s===0)o[t]=n<<15;else{let a=s-127;if(a>15)o[t]=n<<15|31744;else if(a<-14){let _=-14-a;if(_>10)o[t]=n<<15;else{let m=(128|i<<1)>>_>>1;o[t]=n<<15|m&1023}}else{let _=a+15,m=i<<3;o[t]=n<<15|_<<10|m&1023}}}return o.buffer}async function pe(c){let r=await fetch(`${c}/manifest.json`);if(!r.ok)throw new Error(`Failed to load manifest: ${r.status}`);return r.json()}async function xe(c,r,o){let t=r.tensors[o];if(!t)throw new Error(`Tensor not found: ${o}`);let e=`${c}/${t.file}`,s=await(await fetch(e,{headers:{Range:`bytes=${t.offset}-${t.offset+t.size-1}`}})).arrayBuffer();return t.dtype==="f16"?new Uint16Array(s):new Float32Array(s)}async function Sa(c,r,o,t){let e=r[t];if(!e||!("data_offsets"in e))throw new Error(`Tensor not found in safetensors: ${t}`);let[n,s]=e.data_offsets,i=o+n,a=o+s-1,m=await(await fetch(c,{headers:{Range:`bytes=${i}-${a}`}})).arrayBuffer();if(e.dtype==="BF16"){let P=fe(m);return new Uint16Array(P)}else{if(e.dtype==="F16")return new Uint16Array(m);if(e.dtype==="F32"){let P=new Float32Array(m),k=new Uint16Array(P.length);for(let u=0;u<P.length;u++)k[u]=Ue(P[u]);return k}else throw new Error(`Unsupported dtype: ${e.dtype}`)}}function Ue(c){let r=new ArrayBuffer(4);new Float32Array(r)[0]=c;let o=new Uint32Array(r)[0],t=o>>16&32768,e=(o>>23&255)-127+15,n=o>>13&1023;return e<=0?t:e>=31?t|31744:t|e<<10|n}function _e(c,r,o){let t=r.byteLength,e=Math.ceil(t/4)*4,n=c.createBuffer({size:e,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,label:o,mappedAtCreation:!0});return new Uint16Array(n.getMappedRange(0,r.byteLength)).set(r),n.unmap(),n}async function Fa(c,r,o,t,e){let n=Object.entries(o.tensors).filter(([_,m])=>m.component===t).map(([_,m])=>_),s=new Map,i=new Map,a=n.length;for(let _=0;_<n.length;_++){let m=n[_],P=o.tensors[m];e&&e({loaded:_,total:a,component:t,tensor:m});let k=await xe(r,o,m),u=_e(c,k,m);s.set(m,u),i.set(m,{shape:P.shape,buffer:u})}return e&&e({loaded:a,total:a,component:t,tensor:"done"}),{buffers:s,tensors:i}}async function ee(c,r,o,t,e){let n=Object.entries(o.tensors).filter(([k,u])=>u.component===t).map(([k,u])=>k),i=o.tensors[n[0]].file;e&&e({loaded:0,total:n.length,component:t,tensor:`downloading ${i}...`});let a=await fetch(`${r}/${i}`);if(!a.ok)throw new Error(`Failed to download ${i}: ${a.status}`);let _=await a.arrayBuffer(),m=new Map,P=new Map;for(let k=0;k<n.length;k++){let u=n[k],d=o.tensors[u],p=new Uint16Array(_,d.offset,d.size/2),w=_e(c,p,u);m.set(u,w),P.set(u,{shape:d.shape,buffer:w}),e&&(k%20===0||k===n.length-1)&&e({loaded:k+1,total:n.length,component:t,tensor:u})}return{buffers:m,tensors:P}}var $a="voxtral-weights",Ma=1,I="tensors";function le(){return new Promise((c,r)=>{let o=indexedDB.open($a,Ma);o.onupgradeneeded=()=>{let t=o.result;t.objectStoreNames.contains(I)||t.createObjectStore(I)},o.onsuccess=()=>c(o.result),o.onerror=()=>r(o.error)})}async function Ta(c,r){return new Promise((o,t)=>{let s=c.transaction(I,"readonly").objectStore(I).get(r);s.onsuccess=()=>o(s.result??null),s.onerror=()=>t(s.error)})}async function za(c,r,o){return new Promise((t,e)=>{let i=c.transaction(I,"readwrite").objectStore(I).put(o,r);i.onsuccess=()=>t(),i.onerror=()=>e(i.error)})}async function Ea(c){return new Promise((r,o)=>{let n=c.transaction(I,"readonly").objectStore(I).count();n.onsuccess=()=>r(n.result),n.onerror=()=>o(n.error)})}async function Oa(){let c=await le();return new Promise((r,o)=>{let n=c.transaction(I,"readwrite").objectStore(I).clear();n.onsuccess=()=>{c.close(),r()},n.onerror=()=>{c.close(),o(n.error)}})}async function La(){let c=await le();return new Promise((r,o)=>{let n=c.transaction(I,"readonly").objectStore(I).openCursor(),s=0,i=0;n.onsuccess=()=>{let a=n.result;if(a){s++;let _=a.value;(_ instanceof ArrayBuffer||_&&_.byteLength!==void 0)&&(i+=_.byteLength),a.continue()}else c.close(),r({count:s,sizeBytes:i})},n.onerror=()=>{c.close(),o(n.error)}})}var ie="https://huggingface.co/mistralai/Voxtral-4B-TTS-2603/resolve/main/consolidated.safetensors";function Ia(c){return c.startsWith("acoustic_transformer.")?"fm":c.startsWith("audio_tokenizer.")?"codec":c.startsWith("layers.")||c.startsWith("norm.")||c.startsWith("mm_audio_embeddings.")?"backbone":"other"}async function he(c,r=ie,o){let{header:t,dataOffset:e}=await ye(r),n=await le(),s=await Ea(n),i=[];for(let[l,v]of Object.entries(t)){if(l==="__metadata__")continue;let y=v;if(!y.data_offsets)continue;let g=Ia(l);g!=="other"&&i.push({name:l,entry:y,component:g})}let a={backbone:0,fm:1,codec:2};i.sort((l,v)=>(a[l.component]??9)-(a[v.component]??9));let _=i.length,m=0;o&&o({loaded:0,total:_,component:"init",tensor:s>0?`${s} tensors cached in IndexedDB`:"Starting fresh download...",cached:!1,bytesDownloaded:0});let P={backbone:{buffers:new Map,tensors:new Map},fm:{buffers:new Map,tensors:new Map},codec:{buffers:new Map,tensors:new Map}},k=`v1:${e}:${_}`,u=6,d=0;async function p(l,v,y){let g=await Ta(n,y);if(g)return{f16Data:new Uint16Array(g),fromCache:!0,fetchedBytes:0};let[C,U]=v.data_offsets,x=e+C,q=e+U-1,S=U-C,T=await fetch(r,{headers:{Range:`bytes=${x}-${q}`}});if(!T.ok&&T.status!==206)throw new Error(`Failed to fetch tensor ${l}: HTTP ${T.status}`);let $=await T.arrayBuffer(),M;if(v.dtype==="BF16"){let z=fe($);M=new Uint16Array(z)}else if(v.dtype==="F16")M=new Uint16Array($);else if(v.dtype==="F32"){let z=new Float32Array($),E=new Uint16Array(z.length);for(let B=0;B<z.length;B++)E[B]=Ue(z[B]);M=E}else throw new Error(`Unsupported dtype for ${l}: ${v.dtype}`);return await za(n,y,M.buffer),{f16Data:M,fromCache:!1,fetchedBytes:S}}let w=new Map,G=0,h=0,f=new Map;for(;G<i.length&&w.size<u;){let l=G++,{name:v,entry:y}=i[l],g=`${k}:${v}`;w.set(l,p(v,y,g).then(C=>({idx:l,...C})))}for(;h<i.length;){if(f.has(h)){let v=f.get(h);f.delete(h);let{name:y,entry:g,component:C}=i[h];m+=v.fetchedBytes;let U=_e(c,v.f16Data,y),x=P[C];x.buffers.set(y,U),x.tensors.set(y,{shape:g.shape,buffer:U}),d++,o&&o({loaded:d,total:_,component:C,tensor:y,cached:v.fromCache,bytesDownloaded:m}),h++;continue}let l=await Promise.race(w.values());if(w.delete(l.idx),f.set(l.idx,{f16Data:l.f16Data,fromCache:l.fromCache,fetchedBytes:l.fetchedBytes}),G<i.length){let v=G++,{name:y,entry:g}=i[v],C=`${k}:${y}`;w.set(v,p(y,g,C).then(U=>({idx:v,...U})))}}return n.close(),{backbone:P.backbone,fm:P.fm,codec:P.codec}}var K={UNK:0,BOS:1,EOS:2,INST:3,INST_END:4,AUDIO:24,BEGIN_AUDIO:25,OUTPUT_AUDIO:26,AUDIO_TO_TEXT:35,TEXT_TO_AUDIO:36,PAD:11},ge=class c{vocab=new Map;specialTokens=new Map;pattern;voiceNumTokens=new Map;constructor(r){for(let t of r.vocab){let e=atob(t.token_bytes);this.vocab.set(e,t.rank)}let o=r.config.default_num_special_tokens;for(let t of r.special_tokens)this.specialTokens.set(t.token_str,t.rank);try{this.pattern=new RegExp(r.config.pattern,"gu")}catch{this.pattern=/\S+|\s+/gu}if(r.audio?.voice_num_audio_tokens)for(let[t,e]of Object.entries(r.audio.voice_num_audio_tokens))this.voiceNumTokens.set(t,e)}static async load(r){let t=await(await fetch(r)).json();return new c(t)}getVoiceNumTokens(r){let o=this.voiceNumTokens.get(r);if(o===void 0)throw new Error(`Unknown voice: ${r}. Available: ${[...this.voiceNumTokens.keys()].join(", ")}`);return o}buildTTSPrompt(r,o){let t=this.getVoiceNumTokens(o),e=[];e.push(K.BOS),e.push(K.BEGIN_AUDIO);let n=e.length;for(let i=0;i<t;i++)e.push(K.AUDIO);e.push(K.TEXT_TO_AUDIO);let s=this.encode(r);return e.push(...s),e.push(K.AUDIO_TO_TEXT),e.push(K.BEGIN_AUDIO),{tokens:e,audioTokenStart:n,audioTokenCount:t}}encode(r){let o=[],t=r.matchAll(this.pattern);for(let e of t){let n=e[0],s=this.vocab.get(n);if(s!==void 0){o.push(s+1e3);continue}let a=new TextEncoder().encode(n);for(let _ of a){let m=String.fromCharCode(_),P=this.vocab.get(m);P!==void 0?o.push(P+1e3):o.push(K.UNK)}}return o}get voices(){return[...this.voiceNumTokens.keys()]}};var Be=`
struct Params {
  M: u32,
  K: u32,
}

@group(0) @binding(0) var<storage, read> matrix: array<u32>;   // packed f16 [M, K/2]
@group(0) @binding(1) var<storage, read> vector: array<f32>;   // [K]
@group(0) @binding(2) var<storage, read_write> output: array<f32>;  // [M]
@group(0) @binding(3) var<uniform> params: Params;

const WG: u32 = 256u;
var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let row = wid.x;
  if (row >= params.M) { return; }
  let tid = lid.x;

  let K_packed = params.K / 2u;
  var sum: f32 = 0.0;

  // Each thread accumulates over a strided portion of K
  var pk = tid;
  let rowBase = row * K_packed;
  while (pk < K_packed) {
    let packed = matrix[rowBase + pk];
    let pair = unpack2x16float(packed);
    let k0 = pk * 2u;
    sum += f32(pair.x) * vector[k0] + f32(pair.y) * vector[k0 + 1u];
    pk += WG;
  }

  sdata[tid] = sum;
  workgroupBarrier();

  // Tree reduction
  for (var s: u32 = 128u; s > 0u; s >>= 1u) {
    if (tid < s) { sdata[tid] += sdata[tid + s]; }
    workgroupBarrier();
  }

  if (tid == 0u) {
    output[row] = sdata[0];
  }
}
`,Ce=`
struct Params {
  M: u32,
  K: u32,
  row_offset: u32,
}

@group(0) @binding(0) var<storage, read> matrix: array<u32>;   // packed f16 [total_M, K/2]
@group(0) @binding(1) var<storage, read> vector: array<f32>;   // [K]
@group(0) @binding(2) var<storage, read_write> output: array<f32>;  // [total_M]
@group(0) @binding(3) var<uniform> params: Params;

const WG: u32 = 256u;
var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let localRow = wid.x;
  if (localRow >= params.M) { return; }
  let globalRow = localRow + params.row_offset;
  let tid = lid.x;

  let K_packed = params.K / 2u;
  var sum: f32 = 0.0;

  var pk = tid;
  let rowBase = globalRow * K_packed;
  while (pk < K_packed) {
    let packed = matrix[rowBase + pk];
    let pair = unpack2x16float(packed);
    let k0 = pk * 2u;
    sum += f32(pair.x) * vector[k0] + f32(pair.y) * vector[k0 + 1u];
    pk += WG;
  }

  sdata[tid] = sum;
  workgroupBarrier();

  for (var s: u32 = 128u; s > 0u; s >>= 1u) {
    if (tid < s) { sdata[tid] += sdata[tid + s]; }
    workgroupBarrier();
  }

  if (tid == 0u) {
    output[globalRow] = sdata[0];
  }
}
`,qe=`
struct Params {
  dim: u32,
  eps: f32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<u32>;  // packed f16
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const WG: u32 = 256u;
var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;

  // Compute sum of squares
  var ss: f32 = 0.0;
  var i = tid;
  while (i < params.dim) {
    let v = input[i];
    ss += v * v;
    i += WG;
  }
  sdata[tid] = ss;
  workgroupBarrier();

  for (var s: u32 = 128u; s > 0u; s >>= 1u) {
    if (tid < s) { sdata[tid] += sdata[tid + s]; }
    workgroupBarrier();
  }

  let rms = 1.0 / sqrt(sdata[0] / f32(params.dim) + params.eps);

  // Scale and write output
  i = tid;
  while (i < params.dim) {
    let wPacked = weight[i / 2u];
    let wPair = unpack2x16float(wPacked);
    let w = select(f32(wPair.x), f32(wPair.y), (i & 1u) == 1u);
    output[i] = input[i] * rms * w;
    i += WG;
  }
}
`,Ge=`
struct Params {
  token_id: u32,
  dim: u32,
}

@group(0) @binding(0) var<storage, read> table: array<u32>;  // packed f16 [vocab, dim/2]
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.dim) { return; }

  let packedIdx = params.token_id * (params.dim / 2u) + i / 2u;
  let packed = table[packedIdx];
  let pair = unpack2x16float(packed);
  output[i] = select(f32(pair.x), f32(pair.y), (i & 1u) == 1u);
}
`,Ae=`
struct Params {
  dim: u32,       // head_dim (128)
  pos: u32,       // sequence position
  n_heads: u32,   // number of heads to process
  theta: f32,     // rope_theta (1e6 for backbone, 1e4 for FM)
}

@group(0) @binding(0) var<storage, read_write> qk: array<f32>;  // [n_heads, dim]
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total_pairs = params.n_heads * (params.dim / 2u);
  if (idx >= total_pairs) { return; }

  let head = idx / (params.dim / 2u);
  let pair = idx % (params.dim / 2u);

  let freq = 1.0 / pow(params.theta, f32(pair * 2u) / f32(params.dim));
  let angle = f32(params.pos) * freq;
  let cos_a = cos(angle);
  let sin_a = sin(angle);

  let base = head * params.dim + pair * 2u;
  let x0 = qk[base];
  let x1 = qk[base + 1u];

  qk[base]     = x0 * cos_a - x1 * sin_a;
  qk[base + 1u] = x0 * sin_a + x1 * cos_a;
}
`,Se=`
struct Params {
  dim: u32,       // head_dim (128)
  pos: u32,       // sequence position for RoPE angle computation
  n_heads: u32,   // number of heads to process
  theta: f32,     // rope_theta
  offset: u32,    // element offset into qk buffer
}

@group(0) @binding(0) var<storage, read_write> qk: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total_pairs = params.n_heads * (params.dim / 2u);
  if (idx >= total_pairs) { return; }

  let head = idx / (params.dim / 2u);
  let pair = idx % (params.dim / 2u);

  let freq = 1.0 / pow(params.theta, f32(pair * 2u) / f32(params.dim));
  let angle = f32(params.pos) * freq;
  let cos_a = cos(angle);
  let sin_a = sin(angle);

  let base = params.offset + head * params.dim + pair * 2u;
  let x0 = qk[base];
  let x1 = qk[base + 1u];

  qk[base]     = x0 * cos_a - x1 * sin_a;
  qk[base + 1u] = x0 * sin_a + x1 * cos_a;
}
`,Fe=`
struct Params {
  n_heads: u32,
  n_kv_heads: u32,
  head_dim: u32,
  seq_len: u32,      // current position + 1
  kv_repeat: u32,    // n_heads / n_kv_heads
}

@group(0) @binding(0) var<storage, read> q: array<f32>;         // [n_heads, head_dim]
@group(0) @binding(1) var<storage, read> k_cache: array<f32>;   // [max_seq, n_kv_heads * head_dim]
@group(0) @binding(2) var<storage, read_write> scores: array<f32>; // [n_heads, seq_len]
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.n_heads * params.seq_len;
  if (idx >= total) { return; }

  let h = idx / params.seq_len;
  let p = idx % params.seq_len;

  let kv_h = h / params.kv_repeat;
  let scale = 1.0 / sqrt(f32(params.head_dim));

  var dot: f32 = 0.0;
  let qBase = h * params.head_dim;
  let kBase = p * params.n_kv_heads * params.head_dim + kv_h * params.head_dim;

  for (var d: u32 = 0u; d < params.head_dim; d++) {
    dot += q[qBase + d] * k_cache[kBase + d];
  }

  scores[h * params.seq_len + p] = dot * scale;
}
`,$e=`
struct Params {
  n_heads: u32,
  seq_len: u32,
}

@group(0) @binding(0) var<storage, read_write> scores: array<f32>; // [n_heads, seq_len]
@group(0) @binding(1) var<uniform> params: Params;

const WG: u32 = 256u;
var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let h = wid.x;
  if (h >= params.n_heads) { return; }
  let tid = lid.x;
  let base = h * params.seq_len;

  // Find max
  var maxVal: f32 = -1e30;
  var i = tid;
  while (i < params.seq_len) {
    maxVal = max(maxVal, scores[base + i]);
    i += WG;
  }
  sdata[tid] = maxVal;
  workgroupBarrier();
  for (var s: u32 = 128u; s > 0u; s >>= 1u) {
    if (tid < s) { sdata[tid] = max(sdata[tid], sdata[tid + s]); }
    workgroupBarrier();
  }
  let globalMax = sdata[0];

  // Compute exp and sum
  var expSum: f32 = 0.0;
  i = tid;
  while (i < params.seq_len) {
    let e = exp(scores[base + i] - globalMax);
    scores[base + i] = e;
    expSum += e;
    i += WG;
  }
  sdata[tid] = expSum;
  workgroupBarrier();
  for (var s: u32 = 128u; s > 0u; s >>= 1u) {
    if (tid < s) { sdata[tid] += sdata[tid + s]; }
    workgroupBarrier();
  }
  let totalSum = sdata[0];

  // Normalize
  i = tid;
  while (i < params.seq_len) {
    scores[base + i] /= totalSum;
    i += WG;
  }
}
`,Me=`
struct Params {
  n_heads: u32,
  n_kv_heads: u32,
  head_dim: u32,
  seq_len: u32,
  kv_repeat: u32,
}

@group(0) @binding(0) var<storage, read> scores: array<f32>;     // [n_heads, seq_len]
@group(0) @binding(1) var<storage, read> v_cache: array<f32>;    // [max_seq, n_kv_heads * head_dim]
@group(0) @binding(2) var<storage, read_write> output: array<f32>; // [n_heads * head_dim]
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.n_heads * params.head_dim;
  if (idx >= total) { return; }

  let h = idx / params.head_dim;
  let d = idx % params.head_dim;
  let kv_h = h / params.kv_repeat;

  var sum: f32 = 0.0;
  for (var p: u32 = 0u; p < params.seq_len; p++) {
    let score = scores[h * params.seq_len + p];
    let vIdx = p * params.n_kv_heads * params.head_dim + kv_h * params.head_dim + d;
    sum += score * v_cache[vIdx];
  }

  output[idx] = sum;
}
`,Te=`
struct Params {
  pos: u32,
  kv_dim: u32,  // n_kv_heads * head_dim
}

@group(0) @binding(0) var<storage, read> k_new: array<f32>;    // [kv_dim]
@group(0) @binding(1) var<storage, read> v_new: array<f32>;    // [kv_dim]
@group(0) @binding(2) var<storage, read_write> k_cache: array<f32>; // [max_seq, kv_dim]
@group(0) @binding(3) var<storage, read_write> v_cache: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.kv_dim) { return; }

  let cacheIdx = params.pos * params.kv_dim + i;
  k_cache[cacheIdx] = k_new[i];
  v_cache[cacheIdx] = v_new[i];
}
`,ze=`
struct Params {
  dim: u32,
}

@group(0) @binding(0) var<storage, read_write> gate: array<f32>;   // [hidden_dim] from w1 (in-place output)
@group(0) @binding(1) var<storage, read> up: array<f32>;     // [hidden_dim] from w3
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.dim) { return; }

  let g = gate[i];
  let silu = g / (1.0 + exp(-g));
  gate[i] = silu * up[i];
}
`,Ee=`
struct Params {
  dim: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.dim) { return; }
  output[i] = a[i] + b[i];
}
`,Oe=`
struct Params {
  dim: u32,
}

@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.dim) { return; }
  a[i] = a[i] + b[i];
}
`,Le=`
struct Params {
  size: u32,
}

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.size) { return; }
  dst[i] = src[i];
}
`,Ie=`
struct Params {
  dim: u32,
  t: f32,  // timestep value [0, 1]
}

@group(0) @binding(0) var<storage, read_write> output: array<f32>;  // [dim]
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let half_dim = params.dim / 2u;
  if (i >= half_dim) { return; }

  // Match vLLM-Omni/MLX: freq = exp(-log(10000) * i / half_dim), (cos, sin) order
  let freq = exp(-9.210340371976184 * f32(i) / f32(half_dim));
  let angle = params.t * freq;

  // Output layout: [cos_0, cos_1, ..., cos_{n-1}, sin_0, sin_1, ..., sin_{n-1}]
  output[i] = cos(angle);
  output[half_dim + i] = sin(angle);
}
`,We=`
struct Params {
  dim: u32,
  dt: f32,
}

@group(0) @binding(0) var<storage, read_write> x: array<f32>;     // [dim] in-place
@group(0) @binding(1) var<storage, read> velocity: array<f32>;     // [dim]
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.dim) { return; }
  x[i] = x[i] + velocity[i] * params.dt;
}
`,Ne=`
struct Params {
  dim: u32,
  alpha: f32,
}

@group(0) @binding(0) var<storage, read_write> v_cond: array<f32>;  // in-place output
@group(0) @binding(1) var<storage, read> v_uncond: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.dim) { return; }
  v_cond[i] = params.alpha * v_cond[i] + (1.0 - params.alpha) * v_uncond[i];
}
`,De=`
struct Params {
  dim: u32,
  levels: u32,      // 21
  offset: u32,      // 2 (special tokens)
}

@group(0) @binding(0) var<storage, read> input: array<f32>;       // [dim] continuous
@group(0) @binding(1) var<storage, read_write> output: array<u32>; // [dim] quantized codes
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.dim) { return; }

  let clamped = clamp(input[i], -1.0, 1.0);
  let scaled = (clamped + 1.0) * 0.5 * f32(params.levels - 1u);
  output[i] = u32(round(scaled)) + params.offset;
}
`,Re=`
struct Params {
  n_heads: u32,
  n_kv_heads: u32,
  head_dim: u32,
  seq_len: u32,     // 3 for FM
  kv_repeat: u32,
}

@group(0) @binding(0) var<storage, read> q: array<f32>;   // [seq_len * n_heads * head_dim]
@group(0) @binding(1) var<storage, read> k: array<f32>;   // [seq_len * n_kv_heads * head_dim]
@group(0) @binding(2) var<storage, read_write> scores: array<f32>; // [n_heads * seq_len * seq_len]
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.n_heads * params.seq_len * params.seq_len;
  if (idx >= total) { return; }

  let h = idx / (params.seq_len * params.seq_len);
  let rem = idx % (params.seq_len * params.seq_len);
  let qi = rem / params.seq_len;
  let ki = rem % params.seq_len;
  let kv_h = h / params.kv_repeat;

  let scale = 1.0 / sqrt(f32(params.head_dim));
  var dot: f32 = 0.0;

  let qBase = qi * params.n_heads * params.head_dim + h * params.head_dim;
  let kBase = ki * params.n_kv_heads * params.head_dim + kv_h * params.head_dim;

  for (var d: u32 = 0u; d < params.head_dim; d++) {
    dot += q[qBase + d] * k[kBase + d];
  }

  scores[idx] = dot * scale;
}
`,je=`
struct Params {
  n_heads: u32,
  seq_len: u32,  // 3
}

@group(0) @binding(0) var<storage, read_write> scores: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.n_heads * params.seq_len;
  if (idx >= total) { return; }

  let h = idx / params.seq_len;
  let qi = idx % params.seq_len;
  let base = h * params.seq_len * params.seq_len + qi * params.seq_len;

  // Find max
  var maxVal: f32 = -1e30;
  for (var j: u32 = 0u; j < params.seq_len; j++) {
    maxVal = max(maxVal, scores[base + j]);
  }

  // Exp and sum
  var expSum: f32 = 0.0;
  for (var j: u32 = 0u; j < params.seq_len; j++) {
    let e = exp(scores[base + j] - maxVal);
    scores[base + j] = e;
    expSum += e;
  }

  // Normalize
  for (var j: u32 = 0u; j < params.seq_len; j++) {
    scores[base + j] /= expSum;
  }
}
`,Ve=`
struct Params {
  n_heads: u32,
  n_kv_heads: u32,
  head_dim: u32,
  seq_len: u32,
  kv_repeat: u32,
}

@group(0) @binding(0) var<storage, read> scores: array<f32>;
@group(0) @binding(1) var<storage, read> v: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.seq_len * params.n_heads * params.head_dim;
  if (idx >= total) { return; }

  let qi = idx / (params.n_heads * params.head_dim);
  let rem = idx % (params.n_heads * params.head_dim);
  let h = rem / params.head_dim;
  let d = rem % params.head_dim;
  let kv_h = h / params.kv_repeat;

  var sum: f32 = 0.0;
  let scoreBase = h * params.seq_len * params.seq_len + qi * params.seq_len;
  for (var ki: u32 = 0u; ki < params.seq_len; ki++) {
    let score = scores[scoreBase + ki];
    let vIdx = ki * params.n_kv_heads * params.head_dim + kv_h * params.head_dim + d;
    sum += score * v[vIdx];
  }

  output[idx] = sum;
}
`,Ke=`
struct Params {
  n_frames: u32,
  codebook_dim: u32,  // 256
}

@group(0) @binding(0) var<storage, read> codes: array<u32>;     // [n_frames] semantic codes
@group(0) @binding(1) var<storage, read> codebook: array<u32>;  // packed f16 [8192, 256/2]
@group(0) @binding(2) var<storage, read_write> output: array<f32>; // [n_frames, 256]
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.n_frames * params.codebook_dim;
  if (idx >= total) { return; }

  let t = idx / params.codebook_dim;
  let d = idx % params.codebook_dim;

  let code = codes[t];
  let packedIdx = code * (params.codebook_dim / 2u) + d / 2u;
  let packed = codebook[packedIdx];
  let pair = unpack2x16float(packed);
  output[idx] = select(f32(pair.x), f32(pair.y), (d & 1u) == 1u);
}
`,He=`
struct Params {
  n_entries: u32,
  dim: u32,  // 256
  epsilon: f32,
}

@group(0) @binding(0) var<storage, read_write> codebook: array<u32>;  // packed f16 [n_entries, dim/2]
@group(0) @binding(1) var<storage, read> usage: array<u32>;           // packed f16 [n_entries/2]
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.n_entries * (params.dim / 2u);
  if (idx >= total) { return; }

  let entry = idx / (params.dim / 2u);

  // Read usage for this entry (packed f16)
  let usage_pair_idx = entry / 2u;
  let usage_packed = usage[usage_pair_idx];
  let usage_pair = unpack2x16float(usage_packed);
  let usage_val = max(select(f32(usage_pair.x), f32(usage_pair.y), (entry & 1u) == 1u), params.epsilon);

  // Read codebook pair, divide by usage, write back
  let packed = codebook[idx];
  let pair = unpack2x16float(packed);
  let x = f32(pair.x) / usage_val;
  let y = f32(pair.y) / usage_val;
  codebook[idx] = pack2x16float(vec2<f32>(x, y));
}
`,Ye=`
struct Params {
  n_frames: u32,
  n_codebook: u32,   // 36
  levels: u32,        // 21
  offset: u32,        // 2 (special tokens to subtract)
}

@group(0) @binding(0) var<storage, read> codes: array<u32>;       // [n_frames, 36]
@group(0) @binding(1) var<storage, read_write> output: array<f32>; // [n_frames, 36]
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.n_frames * params.n_codebook;
  if (idx >= total) { return; }

  let code = codes[idx] - params.offset;
  output[idx] = f32(code) * 2.0 / f32(params.levels - 1u) - 1.0;
}
`,Qe=`
struct Params {
  c_in: u32,
  c_out: u32,
  kernel: u32,
  n_frames: u32,
  stride: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;    // [n_frames_in, c_in] (time-first)
@group(0) @binding(1) var<storage, read> weight: array<u32>;   // packed f16 [c_out, c_in, kernel]
@group(0) @binding(2) var<storage, read> g: array<u32>;        // packed f16 [c_out, 1, 1] (weight norm scale)
@group(0) @binding(3) var<storage, read_write> output: array<f32>; // [n_frames_out, c_out] (time-first)
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let n_frames_out = params.n_frames;
  let total = params.c_out * n_frames_out;
  if (idx >= total) { return; }

  let co = idx / n_frames_out;
  let t_out = idx % n_frames_out;

  // Get weight norm scale
  let gPacked = g[co / 2u];
  let gPair = unpack2x16float(gPacked);
  let gVal = select(f32(gPair.x), f32(gPair.y), (co & 1u) == 1u);

  // Compute weight L2 norm for this output channel
  var wNorm: f32 = 0.0;
  let wBase = co * params.c_in * params.kernel;
  let wSize = params.c_in * params.kernel;
  let wPacked = wSize / 2u;
  for (var p: u32 = 0u; p < wPacked; p++) {
    let packed = weight[wBase / 2u + p];
    let pair = unpack2x16float(packed);
    wNorm += f32(pair.x) * f32(pair.x) + f32(pair.y) * f32(pair.y);
  }
  wNorm = 1.0 / sqrt(wNorm + 1e-12);

  // Convolution with causal padding \u2014 input is [T, C] layout
  var sum: f32 = 0.0;
  let pad = params.kernel - 1u;
  let n_frames_in = params.n_frames * params.stride;

  for (var ci: u32 = 0u; ci < params.c_in; ci++) {
    for (var k: u32 = 0u; k < params.kernel; k++) {
      let t_in = i32(t_out * params.stride) - i32(pad) + i32(k);
      if (t_in >= 0 && u32(t_in) < n_frames_in) {
        let wIdx = (co * params.c_in + ci) * params.kernel + k;
        let wPacked2 = weight[wIdx / 2u];
        let wPair2 = unpack2x16float(wPacked2);
        let w = select(f32(wPair2.x), f32(wPair2.y), (wIdx & 1u) == 1u);
        // Time-first input: input[t, ci] = input[t * c_in + ci]
        let x = input[u32(t_in) * params.c_in + ci];
        sum += (w * wNorm * gVal) * x;
      }
    }
  }

  // Time-first output: output[t_out, co] = output[t_out * c_out + co]
  output[t_out * params.c_out + co] = sum;
}
`,Xe=`
struct Params {
  c_in: u32,
  c_out: u32,
  kernel: u32,
}

@group(0) @binding(0) var<storage, read> weight: array<u32>;   // packed f16 [c_in, c_out, kernel]
@group(0) @binding(1) var<storage, read> g: array<u32>;        // packed f16 [c_in]
@group(0) @binding(2) var<storage, read_write> scale: array<f32>; // [c_in] output
@group(0) @binding(3) var<uniform> params: Params;

const WG: u32 = 256u;
var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let ci = wid.x;
  if (ci >= params.c_in) { return; }
  let tid = lid.x;

  // Compute ||v[ci, :, :]||^2 with parallel reduction
  let total_elems = params.c_out * params.kernel;
  let base = ci * params.c_out * params.kernel;
  var ss: f32 = 0.0;
  var idx = tid;
  while (idx < total_elems) {
    let wIdx = base + idx;
    let packed = weight[wIdx / 2u];
    let pair = unpack2x16float(packed);
    let w = select(f32(pair.x), f32(pair.y), (wIdx & 1u) == 1u);
    ss += w * w;
    idx += WG;
  }
  sdata[tid] = ss;
  workgroupBarrier();

  for (var s: u32 = 128u; s > 0u; s >>= 1u) {
    if (tid < s) { sdata[tid] += sdata[tid + s]; }
    workgroupBarrier();
  }

  if (tid == 0u) {
    let normInv = 1.0 / sqrt(sdata[0] + 1e-12);
    // Read g[ci]
    let gPacked = g[ci / 2u];
    let gPair = unpack2x16float(gPacked);
    let gVal = select(f32(gPair.x), f32(gPair.y), (ci & 1u) == 1u);
    scale[ci] = gVal * normInv;
  }
}
`,Ze=`
struct Params {
  c_in: u32,
  c_out: u32,
  kernel: u32,
  n_frames_out: u32,  // T_in * stride
  stride: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;    // [T_in, c_in]
@group(0) @binding(1) var<storage, read> weight: array<u32>;   // packed f16 [c_in, c_out, kernel]
@group(0) @binding(2) var<storage, read> scale: array<f32>;    // [c_in] precomputed g[ci]/||v[ci,:,:]||
@group(0) @binding(3) var<storage, read_write> output: array<f32>; // [n_frames_out, c_out]
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.c_out * params.n_frames_out;
  if (idx >= total) { return; }

  let co = idx / params.n_frames_out;
  let t_out = idx % params.n_frames_out;

  // Transposed convolution with per-c_in weight normalization
  var sum: f32 = 0.0;
  let n_frames_in = params.n_frames_out / params.stride;

  for (var k: u32 = 0u; k < params.kernel; k++) {
    let diff = i32(t_out) - i32(k);
    if (diff >= 0 && (u32(diff) % params.stride) == 0u) {
      let t_in = u32(diff) / params.stride;
      if (t_in < n_frames_in) {
        for (var ci: u32 = 0u; ci < params.c_in; ci++) {
          let wIdx = (ci * params.c_out + co) * params.kernel + k;
          let wPacked = weight[wIdx / 2u];
          let wPair = unpack2x16float(wPacked);
          let w = select(f32(wPair.x), f32(wPair.y), (wIdx & 1u) == 1u);
          let x = input[t_in * params.c_in + ci];
          sum += (w * scale[ci]) * x;
        }
      }
    }
  }

  output[t_out * params.c_out + co] = sum;
}
`,Je=`
struct Params {
  dim: u32,
  n_frames: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> scale: array<u32>;  // packed f16 [dim]
@group(0) @binding(2) var<storage, read> residual: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.dim * params.n_frames;
  if (idx >= total) { return; }

  let d = idx % params.dim;
  let sPacked = scale[d / 2u];
  let sPair = unpack2x16float(sPacked);
  let s = select(f32(sPair.x), f32(sPair.y), (d & 1u) == 1u);

  output[idx] = input[idx] * s + residual[idx];
}
`,ea=`
struct Params {
  dim: u32,
}

struct MaxResult {
  value: f32,
  index: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<u32>;  // [1] index
@group(0) @binding(2) var<uniform> params: Params;

const WG: u32 = 256u;

struct SharedEntry {
  value: f32,
  index: u32,
}
var<workgroup> sdata: array<SharedEntry, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;

  // Each thread finds max over its portion
  var bestVal: f32 = -1e30;
  var bestIdx: u32 = 0u;

  var i = tid;
  while (i < params.dim) {
    let v = input[i];
    if (v > bestVal) {
      bestVal = v;
      bestIdx = i;
    }
    i += WG;
  }

  sdata[tid] = SharedEntry(bestVal, bestIdx);
  workgroupBarrier();

  // Reduction
  for (var s: u32 = 128u; s > 0u; s >>= 1u) {
    if (tid < s) {
      if (sdata[tid + s].value > sdata[tid].value) {
        sdata[tid] = sdata[tid + s];
      }
    }
    workgroupBarrier();
  }

  if (tid == 0u) {
    result[0] = sdata[0].index;
  }
}
`,aa=`
struct Params {
  M: u32,
  K: u32,
  src_offset: u32,
  dst_offset: u32,
}

@group(0) @binding(0) var<storage, read> matrix: array<u32>;
@group(0) @binding(1) var<storage, read> vector: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const WG: u32 = 256u;
var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let row = wid.x;
  if (row >= params.M) { return; }
  let tid = lid.x;

  let K_packed = params.K / 2u;
  var sum: f32 = 0.0;

  var pk = tid;
  let rowBase = row * K_packed;
  while (pk < K_packed) {
    let packed = matrix[rowBase + pk];
    let pair = unpack2x16float(packed);
    let k0 = pk * 2u;
    sum += f32(pair.x) * vector[params.src_offset + k0]
         + f32(pair.y) * vector[params.src_offset + k0 + 1u];
    pk += WG;
  }

  sdata[tid] = sum;
  workgroupBarrier();

  for (var s: u32 = 128u; s > 0u; s >>= 1u) {
    if (tid < s) { sdata[tid] += sdata[tid + s]; }
    workgroupBarrier();
  }

  if (tid == 0u) {
    output[params.dst_offset + row] = sdata[0];
  }
}
`,ta=`
struct Params {
  dim: u32,
  eps: f32,
  src_offset: u32,
  dst_offset: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const WG: u32 = 256u;
var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;

  var ss: f32 = 0.0;
  var i = tid;
  while (i < params.dim) {
    let v = input[params.src_offset + i];
    ss += v * v;
    i += WG;
  }
  sdata[tid] = ss;
  workgroupBarrier();

  for (var s: u32 = 128u; s > 0u; s >>= 1u) {
    if (tid < s) { sdata[tid] += sdata[tid + s]; }
    workgroupBarrier();
  }

  let rms = 1.0 / sqrt(sdata[0] / f32(params.dim) + params.eps);

  i = tid;
  while (i < params.dim) {
    let wPacked = weight[i / 2u];
    let wPair = unpack2x16float(wPacked);
    let w = select(f32(wPair.x), f32(wPair.y), (i & 1u) == 1u);
    output[params.dst_offset + i] = input[params.src_offset + i] * rms * w;
    i += WG;
  }
}
`,ra=`
struct Params {
  dim: u32,
  off_a: u32,
  off_b: u32,
  dst_offset: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.dim) { return; }
  output[params.dst_offset + i] = a[params.off_a + i] + b[params.off_b + i];
}
`,sa=`
struct Params {
  dim: u32,
  off_a: u32,
  off_b: u32,
}

@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.dim) { return; }
  a[params.off_a + i] = a[params.off_a + i] + b[params.off_b + i];
}
`,oa=`
struct Params {
  dim: u32,
  off_gate: u32,
  off_up: u32,
}

@group(0) @binding(0) var<storage, read_write> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.dim) { return; }

  let g = gate[params.off_gate + i];
  let silu = g / (1.0 + exp(-g));
  gate[params.off_gate + i] = silu * up[params.off_up + i];
}
`,ia=`
struct Params {
  size: u32,
  src_offset: u32,
  dst_offset: u32,
}

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.size) { return; }
  dst[params.dst_offset + i] = src[params.src_offset + i];
}
`,na=`
struct Params {
  n_heads: u32,
  head_dim: u32,
  seq_len: u32,
  window: u32,      // sliding window size (0 = full attention)
}

@group(0) @binding(0) var<storage, read> q: array<f32>;     // [seq_len, n_heads, head_dim]
@group(0) @binding(1) var<storage, read> k: array<f32>;     // [seq_len, n_heads, head_dim]
@group(0) @binding(2) var<storage, read_write> scores: array<f32>;  // [n_heads, seq_len, seq_len]
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  // gid.x = ki_group (within seq_len), gid.y = qi, gid.z = head
  let ki = gid.x;
  let qi = gid.y;
  let h = gid.z;
  if (ki >= params.seq_len || qi >= params.seq_len || h >= params.n_heads) { return; }
  let idx = h * params.seq_len * params.seq_len + qi * params.seq_len + ki;

  // Causal mask: ki > qi means future \u2014 mask out
  if (ki > qi) {
    scores[idx] = -1e30;
    return;
  }

  // Sliding window mask
  if (params.window > 0u && qi - ki >= params.window) {
    scores[idx] = -1e30;
    return;
  }

  let scale = 1.0 / sqrt(f32(params.head_dim));

  // ALiBi slope: 2^(-8*h/n_heads) \u2014 geometric series from 2^(-8/n) to 2^(-8)
  let slope = pow(2.0, -8.0 * f32(h + 1u) / f32(params.n_heads));
  let alibi = -slope * f32(qi - ki);

  var dot: f32 = 0.0;
  let qBase = qi * params.n_heads * params.head_dim + h * params.head_dim;
  let kBase = ki * params.n_heads * params.head_dim + h * params.head_dim;

  for (var d: u32 = 0u; d < params.head_dim; d++) {
    dot += q[qBase + d] * k[kBase + d];
  }

  scores[idx] = dot * scale + alibi;
}
`,da=`
struct Params {
  n_heads: u32,
  seq_len: u32,
}

@group(0) @binding(0) var<storage, read_write> scores: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.n_heads * params.seq_len;
  if (idx >= total) { return; }

  let h = idx / params.seq_len;
  let qi = idx % params.seq_len;
  let base = h * params.seq_len * params.seq_len + qi * params.seq_len;

  var maxVal: f32 = -1e30;
  for (var j: u32 = 0u; j < params.seq_len; j++) {
    maxVal = max(maxVal, scores[base + j]);
  }

  var expSum: f32 = 0.0;
  for (var j: u32 = 0u; j < params.seq_len; j++) {
    let e = exp(scores[base + j] - maxVal);
    scores[base + j] = e;
    expSum += e;
  }

  for (var j: u32 = 0u; j < params.seq_len; j++) {
    scores[base + j] /= expSum;
  }
}
`,ca=`
struct Params {
  n_heads: u32,
  head_dim: u32,
  seq_len: u32,
}

@group(0) @binding(0) var<storage, read> scores: array<f32>;   // [n_heads, seq_len, seq_len]
@group(0) @binding(1) var<storage, read> v: array<f32>;        // [seq_len, n_heads, head_dim]
@group(0) @binding(2) var<storage, read_write> output: array<f32>; // [seq_len, n_heads, head_dim]
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.seq_len * params.n_heads * params.head_dim;
  if (idx >= total) { return; }

  let qi = idx / (params.n_heads * params.head_dim);
  let rem = idx % (params.n_heads * params.head_dim);
  let h = rem / params.head_dim;
  let d = rem % params.head_dim;

  var sum: f32 = 0.0;
  let scoreBase = h * params.seq_len * params.seq_len + qi * params.seq_len;
  for (var ki: u32 = 0u; ki < params.seq_len; ki++) {
    let score = scores[scoreBase + ki];
    let vIdx = ki * params.n_heads * params.head_dim + h * params.head_dim + d;
    sum += score * v[vIdx];
  }

  output[idx] = sum;
}
`,ua=`
struct Params {
  M: u32,
  K: u32,
  T: u32,  // number of frames
}

@group(0) @binding(0) var<storage, read> matrix: array<u32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const WG: u32 = 256u;
var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let row = wid.x;
  let frame = wid.y;
  if (row >= params.M || frame >= params.T) { return; }
  let tid = lid.x;

  let K_packed = params.K / 2u;
  var sum: f32 = 0.0;

  var pk = tid;
  let rowBase = row * K_packed;
  let inputBase = frame * params.K;
  while (pk < K_packed) {
    let packed = matrix[rowBase + pk];
    let pair = unpack2x16float(packed);
    let k0 = pk * 2u;
    sum += f32(pair.x) * input[inputBase + k0]
         + f32(pair.y) * input[inputBase + k0 + 1u];
    pk += WG;
  }

  sdata[tid] = sum;
  workgroupBarrier();

  for (var s: u32 = 128u; s > 0u; s >>= 1u) {
    if (tid < s) { sdata[tid] += sdata[tid + s]; }
    workgroupBarrier();
  }

  if (tid == 0u) {
    output[frame * params.M + row] = sdata[0];
  }
}
`,ma=`
struct Params {
  dim: u32,
  eps: f32,
  T: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const WG: u32 = 256u;
var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let frame = wid.x;
  if (frame >= params.T) { return; }
  let tid = lid.x;
  let base = frame * params.dim;

  var ss: f32 = 0.0;
  var i = tid;
  while (i < params.dim) {
    let v = input[base + i];
    ss += v * v;
    i += WG;
  }
  sdata[tid] = ss;
  workgroupBarrier();

  for (var s: u32 = 128u; s > 0u; s >>= 1u) {
    if (tid < s) { sdata[tid] += sdata[tid + s]; }
    workgroupBarrier();
  }

  let rms = 1.0 / sqrt(sdata[0] / f32(params.dim) + params.eps);

  i = tid;
  while (i < params.dim) {
    let wPacked = weight[i / 2u];
    let wPair = unpack2x16float(wPacked);
    let w = select(f32(wPair.x), f32(wPair.y), (i & 1u) == 1u);
    output[base + i] = input[base + i] * rms * w;
    i += WG;
  }
}
`,fa=`
struct Params {
  total: u32,  // T * hidden_dim
}

@group(0) @binding(0) var<storage, read_write> gate: array<f32>;  // in-place output
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.total) { return; }
  let g = gate[i];
  let silu = g / (1.0 + exp(-g));
  gate[i] = silu * up[i];
}
`,pa=`
struct Params {
  total: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.total) { return; }
  output[i] = a[i] + b[i];
}
`,_a=`
struct Params {
  total: u32,
}

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.total) { return; }
  dst[i] = src[i];
}
`,la=`
struct Params {
  dim: u32,
  total: u32,  // T * dim
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> scale: array<u32>;  // packed f16 [dim]
@group(0) @binding(2) var<storage, read> residual: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.total) { return; }

  let d = idx % params.dim;
  let sPacked = scale[d / 2u];
  let sPair = unpack2x16float(sPacked);
  let s = select(f32(sPair.x), f32(sPair.y), (d & 1u) == 1u);

  output[idx] = input[idx] * s + residual[idx];
}
`,ha=`
struct Params {
  n_heads: u32,
  head_dim: u32,
  seq_len: u32,
  eps: f32,
}

@group(0) @binding(0) var<storage, read_write> data: array<f32>;  // [seq_len, n_heads * head_dim]
@group(0) @binding(1) var<storage, read> weight: array<u32>;  // packed f16 [n_heads * head_dim]
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.seq_len * params.n_heads;
  if (idx >= total) { return; }

  let base = idx * params.head_dim;
  // Head index for weight lookup: weight is [n_heads * head_dim], each head has head_dim elements
  let h = idx % params.n_heads;
  let wOffset = h * params.head_dim;

  // Compute RMS
  var ss: f32 = 0.0;
  for (var d: u32 = 0u; d < params.head_dim; d++) {
    let v = data[base + d];
    ss += v * v;
  }
  let rms = 1.0 / sqrt(ss / f32(params.head_dim) + params.eps);

  // Normalize with per-head weight (in-place)
  for (var d: u32 = 0u; d < params.head_dim; d++) {
    let wIdx = wOffset + d;
    let wPacked = weight[wIdx / 2u];
    let wPair = unpack2x16float(wPacked);
    let w = select(f32(wPair.x), f32(wPair.y), (wIdx & 1u) == 1u);
    data[base + d] = data[base + d] * rms * w;
  }
}
`,ga=`
struct Params {
  T: u32,
  sem_dim: u32,   // 256
  ac_dim: u32,    // 36
}

@group(0) @binding(0) var<storage, read> semantic: array<f32>;  // [T, 256]
@group(0) @binding(1) var<storage, read> acoustic: array<f32>;  // [T, 36]
@group(0) @binding(2) var<storage, read_write> output: array<f32>;  // [T, 292]
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let out_dim = params.sem_dim + params.ac_dim;
  let total = params.T * out_dim;
  if (idx >= total) { return; }

  let t = idx / out_dim;
  let d = idx % out_dim;

  if (d < params.sem_dim) {
    output[idx] = semantic[t * params.sem_dim + d];
  } else {
    output[idx] = acoustic[t * params.ac_dim + (d - params.sem_dim)];
  }
}
`,ba=`
struct Params {
  dim: u32,
  acoustic_base: u32,    // 8194 (semantic_codebook_size + 2 specials)
  acoustic_stride: u32,  // 23 (acoustic_codebook_size + 2 specials)
  n_acoustic: u32,       // 36
}

@group(0) @binding(0) var<storage, read> table: array<u32>;           // packed f16 [vocab, dim/2]
@group(0) @binding(1) var<storage, read> semantic_code: array<u32>;   // [1] semantic argmax
@group(0) @binding(2) var<storage, read> acoustic_codes: array<u32>;  // [36] FSQ codes with +2 offset
@group(0) @binding(3) var<storage, read_write> output: array<f32>;    // [dim]
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.dim) { return; }

  let half_i = i / 2u;
  let is_odd = (i & 1u) == 1u;
  var sum: f32 = 0.0;

  // Semantic codebook (offset 0)
  let sem_row = semantic_code[0];
  let sem_packed = table[sem_row * (params.dim / 2u) + half_i];
  let sem_pair = unpack2x16float(sem_packed);
  sum += select(f32(sem_pair.x), f32(sem_pair.y), is_odd);

  // 36 acoustic codebooks
  for (var k: u32 = 0u; k < params.n_acoustic; k++) {
    let ac_row = params.acoustic_base + k * params.acoustic_stride + acoustic_codes[k];
    let ac_packed = table[ac_row * (params.dim / 2u) + half_i];
    let ac_pair = unpack2x16float(ac_packed);
    sum += select(f32(ac_pair.x), f32(ac_pair.y), is_odd);
  }

  output[i] = sum;
}
`,va=`
struct Params {
  total: u32,
}

@group(0) @binding(0) var<storage, read_write> buf: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.total) { return; }
  buf[i] = 0.0;
}
`;function b(c,r){return Math.ceil(c/r)}function Na(c,r,o){let t=c.length,e=new Float32Array(t);for(let u=0;u<t;u++)e[u]=c[u]/o;let n=-1/0;for(let u=0;u<t;u++)e[u]>n&&(n=e[u]);let s=0;for(let u=0;u<t;u++)e[u]=Math.exp(e[u]-n),s+=e[u];for(let u=0;u<t;u++)e[u]/=s;let i=Array.from({length:t},(u,d)=>d);i.sort((u,d)=>e[d]-e[u]);let a=0,_=t;for(let u=0;u<t;u++)if(a+=e[i[u]],a>=r){_=u+1;break}let m=0;for(let u=0;u<_;u++)m+=e[i[u]];let P=Math.random()*m,k=0;for(let u=0;u<_;u++)if(k+=e[i[u]],k>=P)return i[u];return i[0]}var be=class{device=null;config;maxSeqLen;modelBuffers=null;workBuffers=null;pipelines=null;kvCaches=[];position=0;constructor(r={}){this.config=r.config||J,this.maxSeqLen=r.maxSeqLen||4096}async init(){if(!navigator.gpu)throw new Error("WebGPU not supported");let r=await navigator.gpu.requestAdapter({powerPreference:"high-performance"});if(!r)throw new Error("No WebGPU adapter");let o=[];r.features.has("shader-f16")&&o.push("shader-f16");let t=2*1024*1024*1024,e=r.limits.maxBufferSize,n=r.limits.maxStorageBufferBindingSize,s=e>=t&&n>=t;this.device=await r.requestDevice({requiredFeatures:o,requiredLimits:{maxBufferSize:s?t:e,maxStorageBufferBindingSize:s?t:n}}),this.createWorkBuffers(),this.createKVCaches(),this.createPipelines()}createPipeline(r,o){let t=this.device,e=t.createShaderModule({code:r,label:o});return t.createComputePipeline({layout:"auto",compute:{module:e,entryPoint:"main"},label:o})}createPipelines(){let r=(o,t)=>this.createPipeline(o,t);this.pipelines={matvecF16:r(Be,"matvecF16"),matvecF16Chunked:r(Ce,"matvecF16Chunked"),matvecF16Offset:r(aa,"matvecF16Offset"),rmsNorm:r(qe,"rmsNorm"),rmsNormOffset:r(ta,"rmsNormOffset"),embeddingLookup:r(Ge,"embeddingLookup"),rope:r(Ae,"rope"),ropeOffset:r(Se,"ropeOffset"),attnScore:r(Fe,"attnScore"),softmax:r($e,"softmax"),attnValue:r(Me,"attnValue"),kvCacheWrite:r(Te,"kvCacheWrite"),swiGLU:r(ze,"swiGLU"),addVectors:r(Ee,"addVectors"),addVectorsOffset:r(ra,"addVectorsOffset"),addInPlace:r(Oe,"addInPlace"),addInPlaceOffset:r(sa,"addInPlaceOffset"),copyBuffer:r(Le,"copyBuffer"),copyBufferOffset:r(ia,"copyBufferOffset"),timeEmbedding:r(Ie,"timeEmbedding"),eulerStep:r(We,"eulerStep"),cfgCombine:r(Ne,"cfgCombine"),fsqQuantize:r(De,"fsqQuantize"),biAttnScore:r(Re,"biAttnScore"),biSoftmax:r(je,"biSoftmax"),biAttnValue:r(Ve,"biAttnValue"),swiGLUOffset:r(oa,"swiGLUOffset"),zeroFill:r(va,"zeroFill"),multiCodebookEmbed:r(ba,"multiCodebookEmbed"),vqLookup:r(Ke,"vqLookup"),fsqDequant:r(Ye,"fsqDequant"),causalConv1d:r(Qe,"causalConv1d"),causalConvTranspose1d:r(Ze,"causalConvTranspose1d"),convTransposeNormScale:r(Xe,"convTransposeNormScale"),layerScale:r(Je,"layerScale"),alibiAttnScore:r(na,"alibiAttnScore"),codecSoftmax:r(da,"codecSoftmax"),codecAttnValue:r(ca,"codecAttnValue"),batchedMatvecF16:r(ua,"batchedMatvecF16"),batchedRmsNorm:r(ma,"batchedRmsNorm"),batchedSwiGLU:r(fa,"batchedSwiGLU"),batchedAdd:r(pa,"batchedAdd"),batchedCopy:r(_a,"batchedCopy"),batchedLayerScale:r(la,"batchedLayerScale"),qkNorm:r(ha,"qkNorm"),concatCodecInput:r(ga,"concatCodecInput"),argmax:r(ea,"argmax"),normalizeCodebook:r(He,"normalizeCodebook")}}createUniform(r){let o=this.device,t=Math.ceil(r.byteLength/16)*16,e=o.createBuffer({size:t,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,mappedAtCreation:!0});return new Uint8Array(e.getMappedRange()).set(new Uint8Array(r)),e.unmap(),e}packUniform(r){let o=new ArrayBuffer(r.length*4),t=new Uint32Array(o),e=new Float32Array(o);for(let n=0;n<r.length;n++){let s=r[n];s.u!==void 0?t[n]=s.u:s.f!==void 0&&(e[n]=s.f)}return this.createUniform(o)}createWorkBuffers(){let r=this.device,o=this.config.backbone,t=this.config.fm,e=GPUBufferUsage.STORAGE,n=GPUBufferUsage.COPY_SRC,s=GPUBufferUsage.COPY_DST,i=(a,_,m=0)=>r.createBuffer({size:a,usage:e|n|s|m,label:_});this.workBuffers={hidden:i(o.dim*4,"hidden"),residual:i(o.dim*4,"residual"),normed:i(o.dim*4,"normed"),q:i(o.n_heads*o.head_dim*4,"q"),k:i(o.n_kv_heads*o.head_dim*4,"k"),v:i(o.n_kv_heads*o.head_dim*4,"v"),attn_out:i(o.n_heads*o.head_dim*4,"attn_out"),scores:i(o.n_heads*this.maxSeqLen*4,"scores"),gate:i(o.hidden_dim*4,"gate"),up:i(o.hidden_dim*4,"up"),down:i(o.dim*4,"down"),x_t:i(t.n_acoustic_out*4,"x_t"),velocity:i(t.n_acoustic_out*4,"velocity"),v_uncond:i(t.n_acoustic_out*4,"v_uncond"),time_embed:i(t.dim*4,"time_embed"),time_proj:i(t.dim*4,"time_proj"),x_t_proj:i(t.dim*4,"x_t_proj"),fm_hidden:i(t.dim*4,"fm_hidden"),fm_residual:i(t.dim*4,"fm_residual"),fm_normed:i(t.dim*4,"fm_normed"),fm_q:i(3*t.n_heads*t.head_dim*4,"fm_q"),fm_k:i(3*t.n_kv_heads*t.head_dim*4,"fm_k"),fm_v:i(3*t.n_kv_heads*t.head_dim*4,"fm_v"),fm_attn_out:i(3*t.n_heads*t.head_dim*4,"fm_attn_out"),fm_scores:i(t.n_heads*3*3*4,"fm_scores"),fm_seq:i(3*t.dim*4,"fm_seq"),fm_gate:i(3*t.hidden_dim*4,"fm_gate"),fm_up:i(3*t.hidden_dim*4,"fm_up"),fm_down:i(3*t.dim*4,"fm_down"),semantic_logits:i(t.semantic_vocab*4,"semantic_logits"),semantic_argmax:i(4,"semantic_argmax"),acoustic_out:i(t.n_acoustic_out*4,"acoustic_out"),acoustic_codes:i(t.n_acoustic_out*4,"acoustic_codes"),logits:i(o.vocab_size*4,"logits"),argmax_result:i(4,"argmax_result")}}createKVCaches(){let r=this.device,o=this.config.backbone,t=o.n_kv_heads*o.head_dim;this.kvCaches=[];for(let e=0;e<o.n_layers;e++)this.kvCaches.push({k:r.createBuffer({size:this.maxSeqLen*t*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,label:`kv_cache.${e}.k`}),v:r.createBuffer({size:this.maxSeqLen*t*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,label:`kv_cache.${e}.v`})})}async loadWeights(r,o){let t=this.device,e=await pe(r),n=o||(()=>{});n({loaded:0,total:3,component:"all",tensor:"Loading backbone..."});let s=await ee(t,r,e,"backbone",o);n({loaded:1,total:3,component:"all",tensor:"Loading FM transformer..."});let i=await ee(t,r,e,"fm",o);n({loaded:2,total:3,component:"all",tensor:"Loading codec decoder..."});let a=await ee(t,r,e,"codec",o);this.modelBuffers=this.organizeWeights(s,i,a),n({loaded:3,total:3,component:"all",tensor:"Done!"})}async loadWeightsFromHF(r=ie,o){let t=this.device,{backbone:e,fm:n,codec:s}=await he(t,r,o);this.modelBuffers=this.organizeWeights(e,n,s),await this.normalizeVQCodebook(),await this.precomputeConvTransposeScales()}async normalizeVQCodebook(){let r=this.device,o=this.pipelines,t=this.modelBuffers,e=this.config.codec,n=this.packUniform([{u:e.semantic_codebook_size},{u:e.semantic_dim},{f:1e-5}]),s=r.createCommandEncoder({label:"normalize_codebook"}),i=s.beginComputePass({label:"normalize_codebook"});this.dispatch(i,o.normalizeCodebook,[t.codec_semantic_codebook,t.codec_cluster_usage,n],[b(e.semantic_codebook_size*e.semantic_dim/2,128)]),i.end(),r.queue.submit([s.finish()]),await r.queue.onSubmittedWorkDone()}async precomputeConvTransposeScales(){let r=this.device,o=this.pipelines,t=this.modelBuffers,e=this.config.codec,n=r.createCommandEncoder({label:"precompute_conv_transpose_scales"});for(let s=0;s<e.decoder_stages;s++){let i=t.codec_stages[s];if(!i.conv_w||!i.conv_g)continue;let a=r.createBuffer({size:e.dim*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,label:`codec_conv_transpose_scale_s${s}`});i.conv_scale=a;let m=this.packUniform([{u:e.dim},{u:e.dim},{u:4}]),P=n.beginComputePass({label:`conv_transpose_norm_scale_s${s}`});this.dispatch(P,o.convTransposeNormScale,[i.conv_w,i.conv_g,a,m],[e.dim]),P.end()}r.queue.submit([n.finish()]),await r.queue.onSubmittedWorkDone()}organizeWeights(r,o,t){let e=(a,_)=>{let m=a.buffers.get(_);if(!m)throw new Error(`Missing weight: ${_}`);return m},n=[];for(let a=0;a<this.config.backbone.n_layers;a++)n.push({attn_norm:e(r,`layers.${a}.attention_norm.weight`),wq:e(r,`layers.${a}.attention.wq.weight`),wk:e(r,`layers.${a}.attention.wk.weight`),wv:e(r,`layers.${a}.attention.wv.weight`),wo:e(r,`layers.${a}.attention.wo.weight`),ffn_norm:e(r,`layers.${a}.ffn_norm.weight`),w1:e(r,`layers.${a}.feed_forward.w1.weight`),w2:e(r,`layers.${a}.feed_forward.w2.weight`),w3:e(r,`layers.${a}.feed_forward.w3.weight`)});let s=[];for(let a=0;a<this.config.fm.n_layers;a++)s.push({attn_norm:e(o,`acoustic_transformer.layers.${a}.attention_norm.weight`),wq:e(o,`acoustic_transformer.layers.${a}.attention.wq.weight`),wk:e(o,`acoustic_transformer.layers.${a}.attention.wk.weight`),wv:e(o,`acoustic_transformer.layers.${a}.attention.wv.weight`),wo:e(o,`acoustic_transformer.layers.${a}.attention.wo.weight`),ffn_norm:e(o,`acoustic_transformer.layers.${a}.ffn_norm.weight`),w1:e(o,`acoustic_transformer.layers.${a}.feed_forward.w1.weight`),w2:e(o,`acoustic_transformer.layers.${a}.feed_forward.w2.weight`),w3:e(o,`acoustic_transformer.layers.${a}.feed_forward.w3.weight`)});let i=[];for(let a=0;a<4;a++){let _=1+a*2,m=2+a*2,P=a<3;i.push({transformer_layers:this.getCodecTransformerLayers(t,_),...P?{conv_w:e(t,`audio_tokenizer.decoder_blocks.${m}.conv.parametrizations.weight.original1`),conv_g:e(t,`audio_tokenizer.decoder_blocks.${m}.conv.parametrizations.weight.original0`)}:{}})}return{tok_embeddings:e(r,"mm_audio_embeddings.tok_embeddings.weight"),audio_embeddings:e(r,"mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight"),backbone_layers:n,final_norm:e(r,"norm.weight"),fm_input_proj:e(o,"acoustic_transformer.input_projection.weight"),fm_llm_proj:e(o,"acoustic_transformer.llm_projection.weight"),fm_time_proj:e(o,"acoustic_transformer.time_projection.weight"),fm_layers:s,fm_norm:e(o,"acoustic_transformer.norm.weight"),fm_semantic_out:e(o,"acoustic_transformer.semantic_codebook_output.weight"),fm_acoustic_out:e(o,"acoustic_transformer.acoustic_codebook_output.weight"),codec_input_conv_w:e(t,"audio_tokenizer.decoder_blocks.0.conv.parametrizations.weight.original1"),codec_input_conv_g:e(t,"audio_tokenizer.decoder_blocks.0.conv.parametrizations.weight.original0"),codec_stages:i,codec_output_conv_w:e(t,"audio_tokenizer.output_proj.conv.parametrizations.weight.original1"),codec_output_conv_g:e(t,"audio_tokenizer.output_proj.conv.parametrizations.weight.original0"),codec_semantic_codebook:e(t,"audio_tokenizer.quantizer.semantic_codebook.embedding_sum"),codec_cluster_usage:e(t,"audio_tokenizer.quantizer.semantic_codebook.cluster_usage")}}getCodecTransformerLayers(r,o){let t=n=>{let s=r.buffers.get(n);if(!s)throw new Error(`Missing codec weight: ${n}`);return s},e=[];for(let n=0;n<2;n++){let s=`audio_tokenizer.decoder_blocks.${o}.layers.${n}`;e.push({attn_norm:t(`${s}.attention_norm.weight`),q_norm:t(`${s}.attention.q_norm.weight`),k_norm:t(`${s}.attention.k_norm.weight`),wq:t(`${s}.attention.wq.weight`),wk:t(`${s}.attention.wk.weight`),wv:t(`${s}.attention.wv.weight`),wo:t(`${s}.attention.wo.weight`),attn_scale:t(`${s}.attention_scale`),ffn_norm:t(`${s}.ffn_norm.weight`),w1:t(`${s}.feed_forward.w1.weight`),w2:t(`${s}.feed_forward.w2.weight`),w3:t(`${s}.feed_forward.w3.weight`),ffn_scale:t(`${s}.ffn_scale`)})}return e}dispatch(r,o,t,e){let n=t.map((i,a)=>({binding:a,resource:{buffer:i}})),s=this.device.createBindGroup({layout:o.getBindGroupLayout(0),entries:n});r.setPipeline(o),r.setBindGroup(0,s),r.dispatchWorkgroups(...e)}backboneStep(r,o,t=!1,e){let n=this.pipelines,s=this.workBuffers,i=this.modelBuffers,a=this.config.backbone,_=this.position,m;m=r.beginComputePass({label:`embed_pos${_}`});let P=this.packUniform([{u:o},{u:a.dim}]),k=t?i.audio_embeddings:i.tok_embeddings;if(this.dispatch(m,n.embeddingLookup,[k,s.hidden,P],[b(a.dim,256)]),m.end(),e){m=r.beginComputePass({label:`voice_embed_pos${_}`});let p=this.packUniform([{u:a.dim}]);this.dispatch(m,n.copyBuffer,[e,s.hidden,p],[b(a.dim,256)]),m.end()}for(let p=0;p<a.n_layers;p++){let w=i.backbone_layers[p],G=this.kvCaches[p];m=r.beginComputePass({label:`layer${p}_attn`});let h=this.packUniform([{u:a.dim}]);this.dispatch(m,n.copyBuffer,[s.hidden,s.residual,h],[b(a.dim,256)]);let f=this.packUniform([{u:a.dim},{f:a.norm_eps}]);this.dispatch(m,n.rmsNorm,[s.hidden,w.attn_norm,s.normed,f],[1]),m.end(),m=r.beginComputePass({label:`layer${p}_qkv`});let l=this.packUniform([{u:a.n_heads*a.head_dim},{u:a.dim}]);this.dispatch(m,n.matvecF16,[w.wq,s.normed,s.q,l],[a.n_heads*a.head_dim]);let v=this.packUniform([{u:a.n_kv_heads*a.head_dim},{u:a.dim}]);this.dispatch(m,n.matvecF16,[w.wk,s.normed,s.k,v],[a.n_kv_heads*a.head_dim]),this.dispatch(m,n.matvecF16,[w.wv,s.normed,s.v,v],[a.n_kv_heads*a.head_dim]),m.end(),m=r.beginComputePass({label:`layer${p}_rope_attn`});let y=this.packUniform([{u:a.head_dim},{u:_},{u:a.n_heads},{f:a.rope_theta}]);this.dispatch(m,n.rope,[s.q,y],[b(a.n_heads*a.head_dim/2,64)]);let g=this.packUniform([{u:a.head_dim},{u:_},{u:a.n_kv_heads},{f:a.rope_theta}]);this.dispatch(m,n.rope,[s.k,g],[b(a.n_kv_heads*a.head_dim/2,64)]);let C=this.packUniform([{u:_},{u:a.n_kv_heads*a.head_dim}]);this.dispatch(m,n.kvCacheWrite,[s.k,s.v,G.k,G.v,C],[b(a.n_kv_heads*a.head_dim,256)]);let U=_+1,x=a.n_heads/a.n_kv_heads,q=this.packUniform([{u:a.n_heads},{u:a.n_kv_heads},{u:a.head_dim},{u:U},{u:x}]);this.dispatch(m,n.attnScore,[s.q,G.k,s.scores,q],[b(a.n_heads*U,64)]),m.end(),m=r.beginComputePass({label:`layer${p}_attn_out`});let S=this.packUniform([{u:a.n_heads},{u:U}]);this.dispatch(m,n.softmax,[s.scores,S],[a.n_heads]);let T=this.packUniform([{u:a.n_heads},{u:a.n_kv_heads},{u:a.head_dim},{u:U},{u:x}]);this.dispatch(m,n.attnValue,[s.scores,G.v,s.attn_out,T],[b(a.n_heads*a.head_dim,128)]),m.end(),m=r.beginComputePass({label:`layer${p}_wo_res`});let $=this.packUniform([{u:a.dim},{u:a.n_heads*a.head_dim}]);this.dispatch(m,n.matvecF16,[w.wo,s.attn_out,s.hidden,$],[a.dim]),m.end(),m=r.beginComputePass({label:`layer${p}_res1`});let M=this.packUniform([{u:a.dim}]);this.dispatch(m,n.addInPlace,[s.hidden,s.residual,M],[b(a.dim,256)]),this.dispatch(m,n.copyBuffer,[s.hidden,s.residual,h],[b(a.dim,256)]);let z=this.packUniform([{u:a.dim},{f:a.norm_eps}]);this.dispatch(m,n.rmsNorm,[s.hidden,w.ffn_norm,s.normed,z],[1]),m.end(),m=r.beginComputePass({label:`layer${p}_ffn`});let E=this.packUniform([{u:a.hidden_dim},{u:a.dim}]);this.dispatch(m,n.matvecF16,[w.w1,s.normed,s.gate,E],[a.hidden_dim]),this.dispatch(m,n.matvecF16,[w.w3,s.normed,s.up,E],[a.hidden_dim]),m.end(),m=r.beginComputePass({label:`layer${p}_ffn_out`});let B=this.packUniform([{u:a.hidden_dim}]);this.dispatch(m,n.swiGLU,[s.gate,s.up,B],[b(a.hidden_dim,256)]);let O=this.packUniform([{u:a.dim},{u:a.hidden_dim}]);this.dispatch(m,n.matvecF16,[w.w2,s.gate,s.hidden,O],[a.dim]),m.end(),m=r.beginComputePass({label:`layer${p}_res2`}),this.dispatch(m,n.addInPlace,[s.hidden,s.residual,M],[b(a.dim,256)]),m.end()}m=r.beginComputePass({label:"final_norm"});let u=this.packUniform([{u:a.dim},{f:a.norm_eps}]);this.dispatch(m,n.rmsNorm,[s.hidden,i.final_norm,s.normed,u],[1]),m.end(),m=r.beginComputePass({label:"lm_head"});for(let w=0;w<a.vocab_size;w+=65535){let G=Math.min(65535,a.vocab_size-w),h=this.packUniform([{u:G},{u:a.dim},{u:w}]);this.dispatch(m,n.matvecF16Chunked,[i.tok_embeddings,s.normed,s.logits,h],[G])}m.end(),m=r.beginComputePass({label:"argmax"});let d=this.packUniform([{u:a.vocab_size}]);this.dispatch(m,n.argmax,[s.logits,s.argmax_result,d],[1]),m.end()}fmTransformerPass(r,o){let t=this.pipelines,e=this.workBuffers,n=this.modelBuffers,s=this.config.fm,i=s.dim,a=3,_=s.n_heads*s.head_dim,m=s.n_kv_heads*s.head_dim,P=s.n_heads/s.n_kv_heads;for(let k=0;k<s.n_layers;k++){let u=n.fm_layers[k],d;d=r.beginComputePass({label:`fm_l${k}_attn_prep`});let p=this.packUniform([{u:a*i},{u:0},{u:0}]);this.dispatch(d,t.copyBufferOffset,[e.fm_seq,e.fm_down,p],[b(a*i,256)]);for(let f=0;f<a;f++){let l=f*i,v=this.packUniform([{u:i},{f:1e-5},{u:l},{u:l}]);this.dispatch(d,t.rmsNormOffset,[e.fm_seq,u.attn_norm,e.fm_gate,v],[1])}d.end(),d=r.beginComputePass({label:`fm_l${k}_qkv`});for(let f=0;f<a;f++){let l=f*i,v=f*_,y=f*m,g=this.packUniform([{u:_},{u:i},{u:l},{u:v}]);this.dispatch(d,t.matvecF16Offset,[u.wq,e.fm_gate,e.fm_q,g],[_]);let C=this.packUniform([{u:m},{u:i},{u:l},{u:y}]);this.dispatch(d,t.matvecF16Offset,[u.wk,e.fm_gate,e.fm_k,C],[m]),this.dispatch(d,t.matvecF16Offset,[u.wv,e.fm_gate,e.fm_v,C],[m])}d.end(),d=r.beginComputePass({label:`fm_l${k}_attn`});let w=this.packUniform([{u:s.n_heads},{u:s.n_kv_heads},{u:s.head_dim},{u:a},{u:P}]);this.dispatch(d,t.biAttnScore,[e.fm_q,e.fm_k,e.fm_scores,w],[b(s.n_heads*a*a,64)]),d.end(),d=r.beginComputePass({label:`fm_l${k}_attn_val`});let G=this.packUniform([{u:s.n_heads},{u:a}]);this.dispatch(d,t.biSoftmax,[e.fm_scores,G],[b(s.n_heads*a,64)]);let h=this.packUniform([{u:s.n_heads},{u:s.n_kv_heads},{u:s.head_dim},{u:a},{u:P}]);this.dispatch(d,t.biAttnValue,[e.fm_scores,e.fm_v,e.fm_attn_out,h],[b(a*s.n_heads*s.head_dim,64)]),d.end(),d=r.beginComputePass({label:`fm_l${k}_wo_res`});for(let f=0;f<a;f++){let l=f*_,v=f*i,y=this.packUniform([{u:i},{u:_},{u:l},{u:v}]);this.dispatch(d,t.matvecF16Offset,[u.wo,e.fm_attn_out,e.fm_seq,y],[i])}d.end(),d=r.beginComputePass({label:`fm_l${k}_res1`});for(let f=0;f<a;f++){let l=f*i,v=this.packUniform([{u:i},{u:l},{u:l}]);this.dispatch(d,t.addInPlaceOffset,[e.fm_seq,e.fm_down,v],[b(i,256)])}this.dispatch(d,t.copyBufferOffset,[e.fm_seq,e.fm_down,p],[b(a*i,256)]),d.end(),d=r.beginComputePass({label:`fm_l${k}_ffn`});for(let f=0;f<a;f++){let l=f*i,v=f*s.hidden_dim,y=this.packUniform([{u:i},{f:1e-5},{u:l},{u:0}]);this.dispatch(d,t.rmsNormOffset,[e.fm_seq,u.ffn_norm,e.fm_normed,y],[1]);let g=this.packUniform([{u:s.hidden_dim},{u:i},{u:0},{u:v}]);this.dispatch(d,t.matvecF16Offset,[u.w1,e.fm_normed,e.fm_gate,g],[s.hidden_dim]),this.dispatch(d,t.matvecF16Offset,[u.w3,e.fm_normed,e.fm_up,g],[s.hidden_dim])}d.end(),d=r.beginComputePass({label:`fm_l${k}_ffn_act`});for(let f=0;f<a;f++){let l=f*s.hidden_dim,v=this.packUniform([{u:s.hidden_dim},{u:l},{u:l}]);this.dispatch(d,t.swiGLUOffset,[e.fm_gate,e.fm_up,v],[b(s.hidden_dim,256)])}d.end(),d=r.beginComputePass({label:`fm_l${k}_ffn_down`});for(let f=0;f<a;f++){let l=f*s.hidden_dim,v=f*i,y=this.packUniform([{u:i},{u:s.hidden_dim},{u:l},{u:v}]);this.dispatch(d,t.matvecF16Offset,[u.w2,e.fm_gate,e.fm_seq,y],[i])}d.end(),d=r.beginComputePass({label:`fm_l${k}_res2`});for(let f=0;f<a;f++){let l=f*i,v=this.packUniform([{u:i},{u:l},{u:l}]);this.dispatch(d,t.addInPlaceOffset,[e.fm_seq,e.fm_down,v],[b(i,256)])}d.end()}{let k=r.beginComputePass({label:"fm_final_norm_vel"}),u=this.packUniform([{u:i},{f:1e-5},{u:0},{u:0}]);this.dispatch(k,t.rmsNormOffset,[e.fm_seq,n.fm_norm,e.fm_normed,u],[1]);let d=this.packUniform([{u:s.n_acoustic_out},{u:i}]);this.dispatch(k,t.matvecF16,[n.fm_acoustic_out,e.fm_normed,o,d],[s.n_acoustic_out]),k.end()}}fmForward(r,o){let t=this.pipelines,e=this.workBuffers,n=this.modelBuffers,s=this.config.fm,i=s.dim,a;a=r.beginComputePass({label:"fm_init"});let _=this.packUniform([{u:s.semantic_vocab},{u:i}]);this.dispatch(a,t.matvecF16,[n.fm_semantic_out,e.normed,e.semantic_logits,_],[s.semantic_vocab]);let m=this.packUniform([{u:i},{u:i}]);this.dispatch(a,t.matvecF16,[n.fm_llm_proj,e.normed,e.fm_hidden,m],[i]);{let u=o??new Float32Array(s.n_acoustic_out);if(!o)for(let d=0;d<s.n_acoustic_out;d++){let p=Math.random(),w=Math.random();u[d]=Math.sqrt(-2*Math.log(p))*Math.cos(2*Math.PI*w)}this.device.queue.writeBuffer(e.x_t,0,u)}a.end(),a=r.beginComputePass({label:"fm_semantic_argmax"});let P=this.packUniform([{u:s.semantic_vocab}]);this.dispatch(a,t.argmax,[e.semantic_logits,e.semantic_argmax,P],[1]),a.end();for(let u=0;u<s.nfe-1;u++){let d=u/(s.nfe-1),p=1/(s.nfe-1);a=r.beginComputePass({label:`fm_step${u}_prep`});let w=this.packUniform([{u:i},{f:d}]);this.dispatch(a,t.timeEmbedding,[e.time_embed,w],[b(i/2,256)]),a.end(),a=r.beginComputePass({label:`fm_step${u}_proj`});let G=this.packUniform([{u:i},{u:i}]);this.dispatch(a,t.matvecF16,[n.fm_time_proj,e.time_embed,e.time_proj,G],[i]);let h=this.packUniform([{u:i},{u:s.n_acoustic_out}]);this.dispatch(a,t.matvecF16,[n.fm_input_proj,e.x_t,e.x_t_proj,h],[i]),a.end(),a=r.beginComputePass({label:`fm_step${u}_assemble`});let f=this.packUniform([{u:i},{u:0},{u:0}]);this.dispatch(a,t.copyBufferOffset,[e.x_t_proj,e.fm_seq,f],[b(i,256)]);let l=this.packUniform([{u:i},{u:0},{u:i}]);this.dispatch(a,t.copyBufferOffset,[e.time_proj,e.fm_seq,l],[b(i,256)]);let v=this.packUniform([{u:i},{u:0},{u:2*i}]);this.dispatch(a,t.copyBufferOffset,[e.fm_hidden,e.fm_seq,v],[b(i,256)]),a.end(),this.fmTransformerPass(r,e.velocity),a=r.beginComputePass({label:`fm_step${u}_uncond`}),this.dispatch(a,t.copyBufferOffset,[e.x_t_proj,e.fm_seq,f],[b(i,256)]),this.dispatch(a,t.copyBufferOffset,[e.time_proj,e.fm_seq,l],[b(i,256)]);let y=this.packUniform([{u:i}]);this.dispatch(a,t.zeroFill,[e.fm_residual,y],[b(i,256)]),this.dispatch(a,t.copyBufferOffset,[e.fm_residual,e.fm_seq,v],[b(i,256)]),a.end(),this.fmTransformerPass(r,e.v_uncond),a=r.beginComputePass({label:`fm_step${u}_euler`});let g=this.packUniform([{u:s.n_acoustic_out},{f:s.cfg_alpha}]);this.dispatch(a,t.cfgCombine,[e.velocity,e.v_uncond,g],[b(s.n_acoustic_out,64)]);let C=this.packUniform([{u:s.n_acoustic_out},{f:p}]);this.dispatch(a,t.eulerStep,[e.x_t,e.velocity,C],[b(s.n_acoustic_out,64)]),a.end()}a=r.beginComputePass({label:"fm_fsq"});let k=this.packUniform([{u:s.n_acoustic_out},{u:this.config.codec.acoustic_codebook_size},{u:2}]);this.dispatch(a,t.fsqQuantize,[e.x_t,e.acoustic_codes,k],[b(s.n_acoustic_out,64)]),a.end()}async codecDecode(r,o){let t=this.device,e=this.pipelines,n=this.modelBuffers,s=this.config.codec,i=r.length,a=s.dim,_=this.uploadArray(r),m=this.uploadArray(o),P=this.createGPUBuffer(i*s.semantic_dim*4,"codec_sem_embed"),k=this.createGPUBuffer(i*s.n_acoustic_codebook*4,"codec_ac_float"),u=this.createGPUBuffer(i*292*4,"codec_concat"),d=i,p=this.createGPUBuffer(d*a*4,"codec_cur"),w=this.createGPUBuffer(d*a*4,"codec_tmp"),G=t.createCommandEncoder({label:"codec_decode"}),h=[],f=(B,O,A,F)=>{let L=G.beginComputePass({label:F});this.dispatch(L,B,O,A),L.end()},l=this.packUniform([{u:i},{u:s.semantic_dim}]);f(e.vqLookup,[_,n.codec_semantic_codebook,P,l],[b(i*s.semantic_dim,128)],"codec_vq");let v=this.packUniform([{u:i},{u:s.n_acoustic_codebook},{u:s.acoustic_codebook_size},{u:2}]);f(e.fsqDequant,[m,k,v],[b(i*s.n_acoustic_codebook,64)],"codec_fsq");let y=this.packUniform([{u:i},{u:s.semantic_dim},{u:s.n_acoustic_codebook}]);f(e.concatCodecInput,[P,k,u,y],[b(i*292,256)],"codec_concat");let g=this.packUniform([{u:292},{u:a},{u:3},{u:d},{u:1}]);f(e.causalConv1d,[u,n.codec_input_conv_w,n.codec_input_conv_g,p,g],[b(a*d,64)],"codec_input_conv");let C=[2,2,2,1],U=[4,4,4,3],x=[2,4,8,16];for(let B=0;B<s.decoder_stages;B++){let O=n.codec_stages[B];for(let A=0;A<s.decoder_layers_per_stage;A++){let F=O.transformer_layers[A],L=d*a*4;w.size<L&&(h.push(w),w=this.createGPUBuffer(L,"codec_tmp"));let W=w,D=d*a,ae=this.packUniform([{u:D}]);f(e.batchedCopy,[p,W,ae],[b(D,256)],`codec_s${B}_l${A}_copy_res`);let Z=this.packUniform([{u:a},{f:s.norm_eps},{u:d}]),H=this.createGPUBuffer(L,"codec_attn_normed");f(e.batchedRmsNorm,[p,F.attn_norm,H,Z],[d],`codec_s${B}_l${A}_attn_norm`);let R=this.createGPUBuffer(d*a*4,"codec_q"),Y=this.createGPUBuffer(d*a*4,"codec_k"),j=this.createGPUBuffer(d*a*4,"codec_v"),N=this.packUniform([{u:a},{u:a},{u:d}]);f(e.batchedMatvecF16,[F.wq,H,R,N],[a,d],`codec_s${B}_l${A}_qproj`),f(e.batchedMatvecF16,[F.wk,H,Y,N],[a,d],`codec_s${B}_l${A}_kproj`),f(e.batchedMatvecF16,[F.wv,H,j,N],[a,d],`codec_s${B}_l${A}_vproj`);let Q=this.packUniform([{u:s.n_heads},{u:s.head_dim},{u:d},{f:s.qk_norm_eps}]);f(e.qkNorm,[R,F.q_norm,Q],[b(d*s.n_heads,128)],`codec_s${B}_l${A}_qnorm`),f(e.qkNorm,[Y,F.k_norm,Q],[b(d*s.n_heads,128)],`codec_s${B}_l${A}_knorm`);let V=this.createGPUBuffer(s.n_heads*d*d*4,"codec_scores"),te=this.packUniform([{u:s.n_heads},{u:s.head_dim},{u:d},{u:x[B]}]);f(e.alibiAttnScore,[R,Y,V,te],[b(d,64),d,s.n_heads],`codec_s${B}_l${A}_attn_score`);let ne=this.packUniform([{u:s.n_heads},{u:d}]);f(e.codecSoftmax,[V,ne],[b(s.n_heads*d,64)],`codec_s${B}_l${A}_softmax`);let X=this.createGPUBuffer(d*a*4,"codec_attn_out"),de=this.packUniform([{u:s.n_heads},{u:s.head_dim},{u:d}]);f(e.codecAttnValue,[V,j,X,de],[b(d*s.n_heads*s.head_dim,64)],`codec_s${B}_l${A}_attn_val`);let ce=this.createGPUBuffer(L,"codec_wo_out");f(e.batchedMatvecF16,[F.wo,X,ce,N],[a,d],`codec_s${B}_l${A}_wo`);let ve=this.packUniform([{u:a},{u:D}]);f(e.batchedLayerScale,[ce,F.attn_scale,W,p,ve],[b(D,256)],`codec_s${B}_l${A}_attn_res`),f(e.batchedCopy,[p,W,ae],[b(D,256)],`codec_s${B}_l${A}_copy_ffn_res`);let re=this.createGPUBuffer(L,"codec_ffn_normed");f(e.batchedRmsNorm,[p,F.ffn_norm,re,Z],[d],`codec_s${B}_l${A}_ffn_norm`);let se=d*s.hidden_dim,oe=this.createGPUBuffer(se*4,"codec_gate"),ue=this.createGPUBuffer(se*4,"codec_up"),ke=this.packUniform([{u:s.hidden_dim},{u:a},{u:d}]);f(e.batchedMatvecF16,[F.w1,re,oe,ke],[s.hidden_dim,d],`codec_s${B}_l${A}_gate`),f(e.batchedMatvecF16,[F.w3,re,ue,ke],[s.hidden_dim,d],`codec_s${B}_l${A}_up`);let wa=this.packUniform([{u:se}]);f(e.batchedSwiGLU,[oe,ue,wa],[b(se,256)],`codec_s${B}_l${A}_swiglu`);let me=this.createGPUBuffer(L,"codec_down"),Pa=this.packUniform([{u:a},{u:s.hidden_dim},{u:d}]);f(e.batchedMatvecF16,[F.w2,oe,me,Pa],[a,d],`codec_s${B}_l${A}_down`),f(e.batchedLayerScale,[me,F.ffn_scale,W,p,ve],[b(D,256)],`codec_s${B}_l${A}_ffn_res`),h.push(H,R,Y,j,V,X,ce,re,oe,ue,me)}if(O.conv_w&&O.conv_scale&&C[B]>1){let A=d*C[B],F=this.createGPUBuffer(A*a*4,"codec_upsampled"),L=this.packUniform([{u:a},{u:a},{u:U[B]},{u:A},{u:C[B]}]);f(e.causalConvTranspose1d,[p,O.conv_w,O.conv_scale,F,L],[b(a*A,64)],`codec_s${B}_conv_up`),h.push(p),p=F,d=A}}let q=d,S=this.createGPUBuffer(q*s.patch_size*4,"codec_output"),T=this.packUniform([{u:a},{u:s.patch_size},{u:7},{u:q},{u:1}]);f(e.causalConv1d,[p,n.codec_output_conv_w,n.codec_output_conv_g,S,T],[b(s.patch_size*q,64)],"codec_output_conv"),t.pushErrorScope("validation"),t.queue.submit([G.finish()]),await t.queue.onSubmittedWorkDone();let $=await t.popErrorScope();$&&(globalThis.__codecError=$.message);let M=q*s.patch_size,z=await this.readF32Array(S,M),E=0;for(let B=0;B<Math.min(z.length,1e3);B++)z[B]!==0&&E++;globalThis.__codecDebug={outT:q,patchSize:s.patch_size,totalSamples:M,nonZero:E,first5:Array.from(z.slice(0,5)),curT:d};for(let B of h)B.destroy();return _.destroy(),m.destroy(),P.destroy(),k.destroy(),u.destroy(),p.destroy(),w.destroy(),S.destroy(),z}async debugCodecDecode(r,o){let t=this.device,e=this.pipelines,n=this.modelBuffers,s=this.config.codec,i=r.length,a=s.dim,_={},m=this.uploadArray(r),P=this.uploadArray(o),k=this.createGPUBuffer(i*s.semantic_dim*4,"codec_sem_embed"),u=this.createGPUBuffer(i*s.n_acoustic_codebook*4,"codec_ac_float"),d=this.createGPUBuffer(i*292*4,"codec_concat"),p=i,w=this.createGPUBuffer(p*a*4,"codec_cur"),G=this.createGPUBuffer(p*a*4,"codec_tmp"),h=[],f=(g,C,U,x,q)=>{let S=g.beginComputePass({label:q});this.dispatch(S,C,U,x),S.end()};{let g=t.createCommandEncoder({label:"codec_phase1"}),C=this.packUniform([{u:i},{u:s.semantic_dim}]);f(g,e.vqLookup,[m,n.codec_semantic_codebook,k,C],[b(i*s.semantic_dim,128)],"codec_vq");let U=this.packUniform([{u:i},{u:s.n_acoustic_codebook},{u:s.acoustic_codebook_size},{u:2}]);f(g,e.fsqDequant,[P,u,U],[b(i*s.n_acoustic_codebook,64)],"codec_fsq");let x=this.packUniform([{u:i},{u:s.semantic_dim},{u:s.n_acoustic_codebook}]);f(g,e.concatCodecInput,[k,u,d,x],[b(i*292,256)],"codec_concat");let q=this.packUniform([{u:292},{u:a},{u:3},{u:p},{u:1}]);f(g,e.causalConv1d,[d,n.codec_input_conv_w,n.codec_input_conv_g,w,q],[b(a*p,64)],"codec_input_conv"),t.queue.submit([g.finish()]),await t.queue.onSubmittedWorkDone()}_.vq_embed=await this.readF32Array(k,i*s.semantic_dim),_.fsq_dequant=await this.readF32Array(u,i*s.n_acoustic_codebook),_.concat=await this.readF32Array(d,i*292),_.after_input_conv=await this.readF32Array(w,p*a);let l=[2,2,2,1],v=[4,4,4,3],y=[2,4,8,16];for(let g=0;g<s.decoder_stages;g++){let C=n.codec_stages[g],U=t.createCommandEncoder({label:`codec_stage${g}`});for(let x=0;x<s.decoder_layers_per_stage;x++){let q=C.transformer_layers[x],S=p*a*4;G.size<S&&(h.push(G),G=this.createGPUBuffer(S,"codec_tmp"));let T=G,$=p*a,M=this.packUniform([{u:$}]);f(U,e.batchedCopy,[w,T,M],[b($,256)],`codec_s${g}_l${x}_copy_res`);let z=this.packUniform([{u:a},{f:s.norm_eps},{u:p}]),E=this.createGPUBuffer(S,"codec_attn_normed");f(U,e.batchedRmsNorm,[w,q.attn_norm,E,z],[p],`codec_s${g}_l${x}_attn_norm`);let B=this.createGPUBuffer(p*a*4,"codec_q"),O=this.createGPUBuffer(p*a*4,"codec_k"),A=this.createGPUBuffer(p*a*4,"codec_v"),F=this.packUniform([{u:a},{u:a},{u:p}]);f(U,e.batchedMatvecF16,[q.wq,E,B,F],[a,p],`codec_s${g}_l${x}_qproj`),f(U,e.batchedMatvecF16,[q.wk,E,O,F],[a,p],`codec_s${g}_l${x}_kproj`),f(U,e.batchedMatvecF16,[q.wv,E,A,F],[a,p],`codec_s${g}_l${x}_vproj`);let L=this.packUniform([{u:s.n_heads},{u:s.head_dim},{u:p},{f:s.qk_norm_eps}]);f(U,e.qkNorm,[B,q.q_norm,L],[b(p*s.n_heads,128)],`codec_s${g}_l${x}_qnorm`),f(U,e.qkNorm,[O,q.k_norm,L],[b(p*s.n_heads,128)],`codec_s${g}_l${x}_knorm`);let W=this.createGPUBuffer(s.n_heads*p*p*4,"codec_scores"),D=this.packUniform([{u:s.n_heads},{u:s.head_dim},{u:p},{u:y[g]}]);f(U,e.alibiAttnScore,[B,O,W,D],[b(p,64),p,s.n_heads],`codec_s${g}_l${x}_attn_score`);let ae=this.packUniform([{u:s.n_heads},{u:p}]);f(U,e.codecSoftmax,[W,ae],[b(s.n_heads*p,64)],`codec_s${g}_l${x}_softmax`);let Z=this.createGPUBuffer(p*a*4,"codec_attn_out"),H=this.packUniform([{u:s.n_heads},{u:s.head_dim},{u:p}]);f(U,e.codecAttnValue,[W,A,Z,H],[b(p*s.n_heads*s.head_dim,64)],`codec_s${g}_l${x}_attn_val`);let R=this.createGPUBuffer(S,"codec_wo_out");f(U,e.batchedMatvecF16,[q.wo,Z,R,F],[a,p],`codec_s${g}_l${x}_wo`);let Y=this.packUniform([{u:a},{u:$}]);f(U,e.batchedLayerScale,[R,q.attn_scale,T,w,Y],[b($,256)],`codec_s${g}_l${x}_attn_res`),f(U,e.batchedCopy,[w,T,M],[b($,256)],`codec_s${g}_l${x}_copy_ffn_res`);let j=this.createGPUBuffer(S,"codec_ffn_normed");f(U,e.batchedRmsNorm,[w,q.ffn_norm,j,z],[p],`codec_s${g}_l${x}_ffn_norm`);let N=p*s.hidden_dim,Q=this.createGPUBuffer(N*4,"codec_gate"),V=this.createGPUBuffer(N*4,"codec_up"),te=this.packUniform([{u:s.hidden_dim},{u:a},{u:p}]);f(U,e.batchedMatvecF16,[q.w1,j,Q,te],[s.hidden_dim,p],`codec_s${g}_l${x}_gate`),f(U,e.batchedMatvecF16,[q.w3,j,V,te],[s.hidden_dim,p],`codec_s${g}_l${x}_up`);let ne=this.packUniform([{u:N}]);f(U,e.batchedSwiGLU,[Q,V,ne],[b(N,256)],`codec_s${g}_l${x}_swiglu`);let X=this.createGPUBuffer(S,"codec_down"),de=this.packUniform([{u:a},{u:s.hidden_dim},{u:p}]);f(U,e.batchedMatvecF16,[q.w2,Q,X,de],[a,p],`codec_s${g}_l${x}_down`),f(U,e.batchedLayerScale,[X,q.ffn_scale,T,w,Y],[b($,256)],`codec_s${g}_l${x}_ffn_res`),h.push(E,B,O,A,W,Z,R,j,Q,V,X)}if(C.conv_w&&C.conv_scale&&l[g]>1){let x=p*l[g],q=this.createGPUBuffer(x*a*4,"codec_upsampled"),S=this.packUniform([{u:a},{u:a},{u:v[g]},{u:x},{u:l[g]}]);f(U,e.causalConvTranspose1d,[w,C.conv_w,C.conv_scale,q,S],[b(a*x,64)],`codec_s${g}_conv_up`),t.queue.submit([U.finish()]),await t.queue.onSubmittedWorkDone(),_[`after_stage${g}_transformer`]=await this.readF32Array(w,p*a),h.push(w),w=q,p=x,_[`after_stage${g}_conv_up`]=await this.readF32Array(w,p*a)}else t.queue.submit([U.finish()]),await t.queue.onSubmittedWorkDone(),_[`after_stage${g}_transformer`]=await this.readF32Array(w,p*a)}{let g=p,C=this.createGPUBuffer(g*s.patch_size*4,"codec_output"),U=this.packUniform([{u:a},{u:s.patch_size},{u:7},{u:g},{u:1}]),x=t.createCommandEncoder({label:"codec_output"});f(x,e.causalConv1d,[w,n.codec_output_conv_w,n.codec_output_conv_g,C,U],[b(s.patch_size*g,64)],"codec_output_conv"),t.queue.submit([x.finish()]),await t.queue.onSubmittedWorkDone(),_.after_output_conv=await this.readF32Array(C,g*s.patch_size),_.audio=_.after_output_conv,h.push(C)}for(let g of h)g.destroy();return m.destroy(),P.destroy(),k.destroy(),u.destroy(),d.destroy(),w.destroy(),G.destroy(),_}uploadArray(r){let o=this.device.createBuffer({size:r.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,mappedAtCreation:!0});return r instanceof Uint32Array?new Uint32Array(o.getMappedRange()).set(r):new Float32Array(o.getMappedRange()).set(r),o.unmap(),o}createGPUBuffer(r,o){return this.device.createBuffer({size:Math.max(r,4),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,label:o})}async readBuffer(r,o){let t=this.device,e=t.createBuffer({size:o,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),n=t.createCommandEncoder();n.copyBufferToBuffer(r,0,e,0,o),t.queue.submit([n.finish()]),await e.mapAsync(GPUMapMode.READ);let s=e.getMappedRange().slice(0);return e.unmap(),e.destroy(),s}async readU32(r){let o=await this.readBuffer(r,4);return new Uint32Array(o)[0]}async readF32Array(r,o){let t=await this.readBuffer(r,o*4);return new Float32Array(t)}async readU32Array(r,o){let t=await this.readBuffer(r,o*4);return new Uint32Array(t)}get isReady(){return this.device!==null&&this.modelBuffers!==null&&this.pipelines!==null}async debugRead(r,o=16){let t=this.workBuffers,e=t[r];if(!e)throw new Error(`Unknown buffer: ${r}. Available: ${Object.keys(t).join(", ")}`);return this.readF32Array(e,o)}async debugBackboneStep(r){let o=this.device.createCommandEncoder();this.backboneStep(o,r),this.device.queue.submit([o.finish()]),await this.device.queue.onSubmittedWorkDone(),this.position++;let t=this.workBuffers,e=await this.readF32Array(t.hidden,16),n=await this.readF32Array(t.normed,16),s=await this.readF32Array(t.logits,16),i=await this.readU32(t.argmax_result),a=await this.readF32Array(t.logits,1024),_=-1/0;for(let m=0;m<a.length;m++)a[m]>_&&(_=a[m]);return{hidden:e,normed:n,logits_first16:s,logits_max:_,argmax:i}}async debugBackboneLayerByLayer(r){let o=this.pipelines,t=this.workBuffers,e=this.modelBuffers,n=this.config.backbone,s=this.position,i=n.dim,a;{let u=this.device.createCommandEncoder();a=u.beginComputePass({label:"debug_embed"});let d=this.packUniform([{u:r},{u:i}]);this.dispatch(a,o.embeddingLookup,[e.tok_embeddings,t.hidden,d],[b(i,256)]),a.end(),this.device.queue.submit([u.finish()]),await this.device.queue.onSubmittedWorkDone()}let _=await this.readF32Array(t.hidden,i),m=[];for(let u=0;u<n.n_layers;u++){let d=e.backbone_layers[u],p=this.kvCaches[u];{let l=this.device.createCommandEncoder();a=l.beginComputePass({label:`debug_l${u}_attn_prep`});let v=this.packUniform([{u:i}]);this.dispatch(a,o.copyBuffer,[t.hidden,t.residual,v],[b(i,256)]);let y=this.packUniform([{u:i},{f:n.norm_eps}]);this.dispatch(a,o.rmsNorm,[t.hidden,d.attn_norm,t.normed,y],[1]),a.end(),this.device.queue.submit([l.finish()]),await this.device.queue.onSubmittedWorkDone()}let w=await this.readF32Array(t.normed,i);{let l=this.device.createCommandEncoder();a=l.beginComputePass({label:`debug_l${u}_qkv`});let v=this.packUniform([{u:n.n_heads*n.head_dim},{u:i}]);this.dispatch(a,o.matvecF16,[d.wq,t.normed,t.q,v],[n.n_heads*n.head_dim]);let y=this.packUniform([{u:n.n_kv_heads*n.head_dim},{u:i}]);this.dispatch(a,o.matvecF16,[d.wk,t.normed,t.k,y],[n.n_kv_heads*n.head_dim]),this.dispatch(a,o.matvecF16,[d.wv,t.normed,t.v,y],[n.n_kv_heads*n.head_dim]),a.end(),a=l.beginComputePass({label:`debug_l${u}_rope_attn`});let g=this.packUniform([{u:n.head_dim},{u:s},{u:n.n_heads},{f:n.rope_theta}]);this.dispatch(a,o.rope,[t.q,g],[b(n.n_heads*n.head_dim/2,64)]);let C=this.packUniform([{u:n.head_dim},{u:s},{u:n.n_kv_heads},{f:n.rope_theta}]);this.dispatch(a,o.rope,[t.k,C],[b(n.n_kv_heads*n.head_dim/2,64)]);let U=this.packUniform([{u:s},{u:n.n_kv_heads*n.head_dim}]);this.dispatch(a,o.kvCacheWrite,[t.k,t.v,p.k,p.v,U],[b(n.n_kv_heads*n.head_dim,256)]);let x=s+1,q=n.n_heads/n.n_kv_heads,S=this.packUniform([{u:n.n_heads},{u:n.n_kv_heads},{u:n.head_dim},{u:x},{u:q}]);this.dispatch(a,o.attnScore,[t.q,p.k,t.scores,S],[b(n.n_heads*x,64)]),a.end(),a=l.beginComputePass({label:`debug_l${u}_attn_out`});let T=this.packUniform([{u:n.n_heads},{u:x}]);this.dispatch(a,o.softmax,[t.scores,T],[n.n_heads]);let $=this.packUniform([{u:n.n_heads},{u:n.n_kv_heads},{u:n.head_dim},{u:x},{u:q}]);this.dispatch(a,o.attnValue,[t.scores,p.v,t.attn_out,$],[b(n.n_heads*n.head_dim,128)]),a.end(),a=l.beginComputePass({label:`debug_l${u}_wo`});let M=this.packUniform([{u:i},{u:n.n_heads*n.head_dim}]);this.dispatch(a,o.matvecF16,[d.wo,t.attn_out,t.hidden,M],[i]),a.end(),a=l.beginComputePass({label:`debug_l${u}_res1`});let z=this.packUniform([{u:i}]);this.dispatch(a,o.addInPlace,[t.hidden,t.residual,z],[b(i,256)]),a.end(),this.device.queue.submit([l.finish()]),await this.device.queue.onSubmittedWorkDone()}let G=await this.readF32Array(t.hidden,i);{let l=this.device.createCommandEncoder();a=l.beginComputePass({label:`debug_l${u}_ffn_prep`});let v=this.packUniform([{u:i}]);this.dispatch(a,o.copyBuffer,[t.hidden,t.residual,v],[b(i,256)]);let y=this.packUniform([{u:i},{f:n.norm_eps}]);this.dispatch(a,o.rmsNorm,[t.hidden,d.ffn_norm,t.normed,y],[1]),a.end(),this.device.queue.submit([l.finish()]),await this.device.queue.onSubmittedWorkDone()}let h=await this.readF32Array(t.normed,i);{let l=this.device.createCommandEncoder();a=l.beginComputePass({label:`debug_l${u}_ffn`});let v=this.packUniform([{u:n.hidden_dim},{u:i}]);this.dispatch(a,o.matvecF16,[d.w1,t.normed,t.gate,v],[n.hidden_dim]),this.dispatch(a,o.matvecF16,[d.w3,t.normed,t.up,v],[n.hidden_dim]),a.end(),a=l.beginComputePass({label:`debug_l${u}_ffn_out`});let y=this.packUniform([{u:n.hidden_dim}]);this.dispatch(a,o.swiGLU,[t.gate,t.up,y],[b(n.hidden_dim,256)]);let g=this.packUniform([{u:i},{u:n.hidden_dim}]);this.dispatch(a,o.matvecF16,[d.w2,t.gate,t.hidden,g],[i]),a.end(),a=l.beginComputePass({label:`debug_l${u}_res2`});let C=this.packUniform([{u:i}]);this.dispatch(a,o.addInPlace,[t.hidden,t.residual,C],[b(i,256)]),a.end(),this.device.queue.submit([l.finish()]),await this.device.queue.onSubmittedWorkDone()}let f=await this.readF32Array(t.hidden,i);m.push({attn_norm:w,attn_out:G,ffn_norm:h,ffn_out:f})}{let u=this.device.createCommandEncoder();a=u.beginComputePass({label:"debug_final_norm"});let d=this.packUniform([{u:i},{f:n.norm_eps}]);this.dispatch(a,o.rmsNorm,[t.hidden,e.final_norm,t.normed,d],[1]),a.end(),this.device.queue.submit([u.finish()]),await this.device.queue.onSubmittedWorkDone()}let P=await this.readF32Array(t.normed,i),k=await this.readF32Array(t.hidden,i);return this.position++,{embed:_,layers:m,final_norm:P,hidden:k}}async debugFMForward(r=42){let o=this.config.fm,t=this.device.createCommandEncoder();this.fmForward(t),this.device.queue.submit([t.finish()]),await this.device.queue.onSubmittedWorkDone();let e=this.workBuffers,n=await this.readF32Array(e.semantic_logits,o.semantic_vocab),s=await this.readU32Array(e.acoustic_codes,o.n_acoustic_out),i=await this.readF32Array(e.x_t,o.n_acoustic_out);return{semantic_logits:n,velocities:[],acoustic_codes:s,x_final:i}}reset(){this.position=0}async backboneStepAndRead(r,o=!1){let t=this.device.createCommandEncoder();this.backboneStep(t,r,o),this.device.queue.submit([t.finish()]),await this.device.queue.onSubmittedWorkDone();let e=await this.readU32(this.workBuffers.argmax_result);return this.position++,e}async debugBackboneStepFull(r,o=!1){let t=this.device.createCommandEncoder();this.backboneStep(t,r,o),this.device.queue.submit([t.finish()]),await this.device.queue.onSubmittedWorkDone();let e=await this.readF32Array(this.workBuffers.normed,this.config.backbone.dim);return this.position++,e}async debugFMStep(r){let o=this.config.fm,t=this.device.createCommandEncoder();this.fmForward(t,r),this.device.queue.submit([t.finish()]),await this.device.queue.onSubmittedWorkDone();let e=this.workBuffers;return{semantic_logits:await this.readF32Array(e.semantic_logits,o.semantic_vocab),acoustic_codes:await this.readU32Array(e.acoustic_codes,o.n_acoustic_out),x_final:await this.readF32Array(e.x_t,o.n_acoustic_out)}}async fmStepAndRead(){let r=this.device.createCommandEncoder();return this.fmForward(r),this.device.queue.submit([r.finish()]),await this.device.queue.onSubmittedWorkDone(),this.readU32Array(this.workBuffers.acoustic_codes,this.config.fm.n_acoustic_out)}async generate(r,o,t,e,n=500,s){if(!this.isReady)throw new Error("Engine not initialized. Call init() and loadWeights() first.");this.reset();let i=performance.now(),a=[];if(e&&t>0){let l=this.config.backbone.dim;for(let v=0;v<t;v++){let y=this.device.createBuffer({size:l*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,mappedAtCreation:!0});new Float32Array(y.getMappedRange()).set(e.subarray(v*l,(v+1)*l)),y.unmap(),a.push(y)}}for(let l=0;l<r.length;l++){let v=r[l],y=this.device.createCommandEncoder();if(l>=o&&l<o+t&&a.length>0){let g=l-o;this.backboneStep(y,v,!1,a[g])}else this.backboneStep(y,v);this.device.queue.submit([y.finish()]),await this.device.queue.onSubmittedWorkDone(),this.position++}let _=performance.now();{let l=this.device.createCommandEncoder();this.backboneStep(l,24,!1),this.device.queue.submit([l.finish()]),await this.device.queue.onSubmittedWorkDone(),this.position++}let m=[],P=[],k=this.config.backbone,u=this.config.fm,d=this.pipelines,p=this.workBuffers,w=this.modelBuffers;for(let l=0;l<n;l++){if(l>0){let q=this.device.createCommandEncoder(),S=q.beginComputePass({label:`multiCBEmbed_frame${l}`}),T=this.packUniform([{u:k.dim},{u:8194},{u:23},{u:36}]);this.dispatch(S,d.multiCodebookEmbed,[w.audio_embeddings,p.semantic_argmax,p.acoustic_codes,p.hidden,T],[b(k.dim,256)]),S.end();let $=q.beginComputePass({label:`mcb_copy_frame${l}`}),M=this.packUniform([{u:k.dim}]);this.dispatch($,d.copyBuffer,[p.hidden,p.fm_gate,M],[b(k.dim,256)]),$.end(),this.backboneStep(q,0,!1,p.fm_gate),this.device.queue.submit([q.finish()]),await this.device.queue.onSubmittedWorkDone(),this.position++}let v=this.device.createCommandEncoder();this.fmForward(v),this.device.queue.submit([v.finish()]),await this.device.queue.onSubmittedWorkDone();let y=await this.readF32Array(p.semantic_logits,u.semantic_vocab);y[0]=-1/0;let g=8194;for(let q=g;q<y.length;q++)y[q]=-1/0;let C=Na(y,.9,.8);if(C<=1)break;m.push(C);let U=new Uint32Array([C]);this.device.queue.writeBuffer(p.semantic_argmax,0,U);let x=await this.readU32Array(p.acoustic_codes,u.n_acoustic_out);P.push(Array.from(x)),s?.(l,C,x)}let G=performance.now(),h;if(m.length>0){let l=new Uint32Array(m),v=new Uint32Array(P.flat());h=await this.codecDecode(l,v)}else h=new Float32Array(0);let f=performance.now();return{semanticCodes:m,acousticCodes:P,audio:h,stats:{backboneMs:_-i,fmMs:G-_,codecMs:f-G,totalMs:f-i,framesGenerated:m.length}}}destroy(){if(this.workBuffers)for(let r of Object.values(this.workBuffers))r.destroy();for(let r of this.kvCaches)r.k.destroy(),r.v.destroy();this.device?.destroy()}};function ka(c){let r=new DataView(c),o=new Uint8Array(c);if(o[0]!==147||o[1]!==78||o[2]!==85||o[3]!==77||o[4]!==80||o[5]!==89)throw new Error("Not a valid .npy file");let t=o[6],e=o[7],n,s;if(t===1)n=r.getUint16(8,!0),s=10;else if(t===2)n=r.getUint32(8,!0),s=12;else throw new Error(`Unsupported npy version: ${t}.${e}`);let i=new TextDecoder().decode(o.slice(s,s+n)),a=i.match(/'descr'\s*:\s*'([^']+)'/),_=i.match(/'shape'\s*:\s*\(([^)]*)\)/),m=i.match(/'fortran_order'\s*:\s*(True|False)/);if(!a||!_)throw new Error(`Cannot parse npy header: ${i}`);let P=a[1],k=_[1].trim(),u=k===""?[]:k.split(",").filter(h=>h.trim()!=="").map(h=>parseInt(h.trim()));if(m?m[1]==="True":!1)throw new Error("Fortran-order arrays not supported");let p=s+n,w=c.slice(p),G;switch(P){case"<f4":case"=f4":G=new Float32Array(w);break;case"<f8":case"=f8":{let h=new Float64Array(w);G=new Float32Array(h.length);for(let f=0;f<h.length;f++)G[f]=h[f];break}case"<i4":case"=i4":G=new Int32Array(w);break;case"<u4":case"=u4":G=new Uint32Array(w);break;case"<i8":case"=i8":{let h=new BigInt64Array(w);G=new Int32Array(h.length);for(let f=0;f<h.length;f++)G[f]=Number(h[f]);break}default:throw new Error(`Unsupported dtype: ${P}`)}return{dtype:P,shape:u,data:G}}async function Da(c){let r=await fetch(c);if(!r.ok)throw new Error(`Failed to fetch ${c}: ${r.status}`);let o=await r.arrayBuffer();return ka(o)}function Ra(c,r,o=.01,t=.01){if(c.length!==r.length)return{passed:!1,maxAbsDiff:1/0,maxRelDiff:1/0,mismatchCount:c.length,totalCount:r.length};let e=0,n=0,s=0;for(let i=0;i<c.length;i++){let a=c[i],_=r[i],m=Math.abs(a-_),P=Math.abs(_)>1e-8?m/Math.abs(_):0;e=Math.max(e,m),n=Math.max(n,P),m>o+t*Math.abs(_)&&s++}return{passed:s===0,maxAbsDiff:e,maxRelDiff:n,mismatchCount:s,totalCount:c.length}}export{ie as HF_VOXTRAL_URL,K as TOKENS,ge as TekkenTokenizer,be as VoxtralEngine,Ra as allclose,Oa as clearWeightCache,fe as convertBF16toF16,J as defaultConfig,La as getWeightCacheInfo,ee as loadComponentBulk,Fa as loadComponentWeights,pe as loadManifest,Da as loadNpy,xe as loadTensorFromManifest,Sa as loadTensorFromSafetensors,he as loadWeightsFromHF,ka as parseNpy,ye as parseSafetensorsHeader,Aa as runBenchmark};
