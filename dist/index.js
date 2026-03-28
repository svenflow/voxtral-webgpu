var xe={backbone:{dim:3072,n_layers:26,head_dim:128,hidden_dim:9216,n_heads:32,n_kv_heads:8,vocab_size:131072,rope_theta:1e6,norm_eps:1e-5},fm:{input_dim:3072,dim:3072,n_layers:3,head_dim:128,hidden_dim:9216,n_heads:32,n_kv_heads:8,nfe:8,cfg_alpha:1.2,rope_theta:1e4,sigma:1e-5,sigma_max:1,n_acoustic_out:36,semantic_vocab:8320},codec:{dim:1024,hidden_dim:4096,head_dim:128,n_heads:8,n_kv_heads:8,semantic_codebook_size:8192,semantic_dim:256,n_acoustic_codebook:36,acoustic_codebook_size:21,sampling_rate:24e3,frame_rate:12.5,patch_size:240,decoder_stages:4,decoder_layers_per_stage:2,decoder_conv_strides:[1,2,2,2],decoder_conv_kernels:[3,4,4,4],attn_sliding_window:16,norm_eps:.01,qk_norm_eps:1e-6,qk_norm:!0,layer_scale:!0,weight_norm_conv:!0}};async function qa(x){let i=await(await fetch(x,{headers:{Range:"bytes=0-7"}})).arrayBuffer(),t=Number(new DataView(i).getBigUint64(0,!0)),n=await(await fetch(x,{headers:{Range:`bytes=8-${8+t-1}`}})).text();return{header:JSON.parse(n),dataOffset:8+t}}function Ca(x){let s=new Uint16Array(x),i=new Uint16Array(s.length);for(let t=0;t<s.length;t++){let a=s[t],n=a>>15&1,r=a>>7&255,o=a&127;if(r===255)i[t]=n<<15|31744|(o?512:0);else if(r===0)i[t]=n<<15;else{let e=r-127;if(e>15)i[t]=n<<15|31744;else if(e<-14){let g=-14-e;if(g>10)i[t]=n<<15;else{let u=(128|o<<1)>>g>>1;i[t]=n<<15|u&1023}}else{let g=e+15,u=o<<3;i[t]=n<<15|g<<10|u&1023}}}return i.buffer}async function Ue(x){let s=await fetch(`${x}/manifest.json`);if(!s.ok)throw new Error(`Failed to load manifest: ${s.status}`);return s.json()}function Ga(x){let s=new ArrayBuffer(4);new Float32Array(s)[0]=x;let i=new Uint32Array(s)[0],t=i>>16&32768,a=(i>>23&255)-127+15,n=i>>13&1023;return a<=0?t:a>=31?t|31744:t|a<<10|n}function Be(x,s,i){let t=s.byteLength,a=Math.ceil(t/4)*4,n=x.createBuffer({size:a,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,label:i,mappedAtCreation:!0});return new Uint16Array(n.getMappedRange(0,s.byteLength)).set(s),n.unmap(),n}async function ue(x,s,i,t,a){let n=Object.entries(i.tensors).filter(([P,c])=>c.component===t).map(([P,c])=>P),o=i.tensors[n[0]].file;a&&a({loaded:0,total:n.length,component:t,tensor:`downloading ${o}...`});let e=await fetch(`${s}/${o}`);if(!e.ok)throw new Error(`Failed to download ${o}: ${e.status}`);let g=await e.arrayBuffer(),u=new Map,q=new Map;for(let P=0;P<n.length;P++){let c=n[P],d=i.tensors[c],f=new Uint16Array(g,d.offset,d.size/2),b=Be(x,f,c);u.set(c,b),q.set(c,{shape:d.shape,buffer:b}),a&&(P%20===0||P===n.length-1)&&a({loaded:P+1,total:n.length,component:t,tensor:c})}return{buffers:u,tensors:q}}var Sa="voxtral-weights",Fa=1,D="tensors";function be(){return new Promise((x,s)=>{let i=indexedDB.open(Sa,Fa);i.onupgradeneeded=()=>{let t=i.result;t.objectStoreNames.contains(D)||t.createObjectStore(D)},i.onsuccess=()=>x(i.result),i.onerror=()=>s(i.error)})}async function Aa(x,s){return new Promise((i,t)=>{let r=x.transaction(D,"readonly").objectStore(D).get(s);r.onsuccess=()=>i(r.result??null),r.onerror=()=>t(r.error)})}async function $a(x,s,i){return new Promise((t,a)=>{let o=x.transaction(D,"readwrite").objectStore(D).put(i,s);o.onsuccess=()=>t(),o.onerror=()=>a(o.error)})}async function Ta(x){return new Promise((s,i)=>{let n=x.transaction(D,"readonly").objectStore(D).count();n.onsuccess=()=>s(n.result),n.onerror=()=>i(n.error)})}async function za(){let x=await be();return new Promise((s,i)=>{let n=x.transaction(D,"readwrite").objectStore(D).clear();n.onsuccess=()=>{x.close(),s()},n.onerror=()=>{x.close(),i(n.error)}})}async function Ma(){let x=await be();return new Promise((s,i)=>{let n=x.transaction(D,"readonly").objectStore(D).openCursor(),r=0,o=0;n.onsuccess=()=>{let e=n.result;if(e){r++;let g=e.value;(g instanceof ArrayBuffer||g&&g.byteLength!==void 0)&&(o+=g.byteLength),e.continue()}else x.close(),s({count:r,sizeBytes:o})},n.onerror=()=>{x.close(),i(n.error)}})}var re="https://huggingface.co/mistralai/Voxtral-4B-TTS-2603/resolve/main/consolidated.safetensors";function Oa(x){return x.startsWith("acoustic_transformer.")?"fm":x.startsWith("audio_tokenizer.")?"codec":x.startsWith("layers.")||x.startsWith("norm.")||x.startsWith("mm_audio_embeddings.")?"backbone":"other"}async function qe(x,s=re,i){let{header:t,dataOffset:a}=await qa(s),n=await be(),r=await Ta(n),o=[];for(let[h,l]of Object.entries(t)){if(h==="__metadata__")continue;let w=l;if(!w.data_offsets)continue;let p=Oa(h);p!=="other"&&o.push({name:h,entry:w,component:p})}let e={backbone:0,fm:1,codec:2};o.sort((h,l)=>(e[h.component]??9)-(e[l.component]??9));let g=o.length,u=0;i&&i({loaded:0,total:g,component:"init",tensor:r>0?`${r} tensors cached in IndexedDB`:"Starting fresh download...",cached:!1,bytesDownloaded:0});let q={backbone:{buffers:new Map,tensors:new Map},fm:{buffers:new Map,tensors:new Map},codec:{buffers:new Map,tensors:new Map}},P=`v1:${a}:${g}`,c=6,d=0;async function f(h,l,w){let p=await Aa(n,w);if(p)return{f16Data:new Uint16Array(p),fromCache:!0,fetchedBytes:0};let[B,k]=l.data_offsets,v=a+B,S=a+k-1,F=k-B,T=await fetch(s,{headers:{Range:`bytes=${v}-${S}`}});if(!T.ok&&T.status!==206)throw new Error(`Failed to fetch tensor ${h}: HTTP ${T.status}`);let M=await T.arrayBuffer(),O;if(l.dtype==="BF16"){let z=Ca(M);O=new Uint16Array(z)}else if(l.dtype==="F16")O=new Uint16Array(M);else if(l.dtype==="F32"){let z=new Float32Array(M),E=new Uint16Array(z.length);for(let L=0;L<z.length;L++)E[L]=Ga(z[L]);O=E}else throw new Error(`Unsupported dtype for ${h}: ${l.dtype}`);return await $a(n,w,O.buffer),{f16Data:O,fromCache:!1,fetchedBytes:F}}let b=new Map,G=0,C=0,m=new Map;for(;G<o.length&&b.size<c;){let h=G++,{name:l,entry:w}=o[h],p=`${P}:${l}`;b.set(h,f(l,w,p).then(B=>({idx:h,...B})))}for(;C<o.length;){if(m.has(C)){let l=m.get(C);m.delete(C);let{name:w,entry:p,component:B}=o[C];u+=l.fetchedBytes;let k=Be(x,l.f16Data,w),v=q[B];v.buffers.set(w,k),v.tensors.set(w,{shape:p.shape,buffer:k}),d++,i&&i({loaded:d,total:g,component:B,tensor:w,cached:l.fromCache,bytesDownloaded:u}),C++;continue}let h=await Promise.race(b.values());if(b.delete(h.idx),m.set(h.idx,{f16Data:h.f16Data,fromCache:h.fromCache,fetchedBytes:h.fetchedBytes}),G<o.length){let l=G++,{name:w,entry:p}=o[l],B=`${P}:${w}`;b.set(l,f(w,p,B).then(k=>({idx:l,...k})))}}return n.close(),{backbone:q.backbone,fm:q.fm,codec:q.codec}}var Ce=`
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
`,Ge=`
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
`,Se=`
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
`,Fe=`
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
`,$e=`
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
`,Te=`
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
`,ze=`
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
`,Oe=`
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
`,We=`
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
`,Le=`
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
`,De=`
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
`,Re=`
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
`,Ve=`
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
  let scaled = round((clamped + 1.0) * 0.5 * f32(params.levels - 1u));
  output[i] = u32(clamp(scaled, 0.0, f32(params.levels - 1u))) + params.offset;
}
`,je=`
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
`,Ke=`
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
`,He=`
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
`,Ye=`
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
`,Qe=`
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
`,Xe=`
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
`,Ze=`
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
  // 2D dispatch: gid.x = t_out (within workgroups), gid.y = co
  let t_out = gid.x;
  let co = gid.y;
  let n_frames_out = params.n_frames;
  if (t_out >= n_frames_out || co >= params.c_out) { return; }

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
`,Je=`
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
`,ea=`
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
  // 2D dispatch: gid.x = t_out (within workgroups), gid.y = co
  let t_out = gid.x;
  let co = gid.y;
  if (t_out >= params.n_frames_out || co >= params.c_out) { return; }

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
`,aa=`
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
`,ta=`
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
`,ra=`
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
`,sa=`
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
`,oa=`
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
`,ia=`
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
`,na=`
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
`,da=`
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
`,ca=`
struct Params {
  n_heads: u32,
  head_dim: u32,
  seq_len: u32,
  window: u32,      // sliding window size (must be > 0)
}

@group(0) @binding(0) var<storage, read> q: array<f32>;     // [seq_len, n_heads, head_dim]
@group(0) @binding(1) var<storage, read> k: array<f32>;     // [seq_len, n_heads, head_dim]
@group(0) @binding(2) var<storage, read_write> scores: array<f32>;  // [n_heads, seq_len, window]
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  // gid.x = slot (within window), gid.y = qi, gid.z = head
  let slot = gid.x;
  let qi = gid.y;
  let h = gid.z;
  let W = params.window;
  if (slot >= W || qi >= params.seq_len || h >= params.n_heads) { return; }

  let idx = h * params.seq_len * W + qi * W + slot;

  // Map slot to actual key position
  // ki_start = max(0, qi - W + 1), ki = ki_start + slot
  var ki_start: u32 = 0u;
  if (qi + 1u > W) { ki_start = qi - W + 1u; }
  let ki = ki_start + slot;

  // Inactive slot (beyond causal boundary or past qi)
  if (ki > qi) {
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
`,ua=`
struct Params {
  n_heads: u32,
  seq_len: u32,
  window: u32,
}

@group(0) @binding(0) var<storage, read_write> scores: array<f32>;  // [n_heads, seq_len, window]
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.n_heads * params.seq_len;
  if (idx >= total) { return; }

  let h = idx / params.seq_len;
  let qi = idx % params.seq_len;
  let W = params.window;
  let base = h * params.seq_len * W + qi * W;

  var maxVal: f32 = -1e30;
  for (var j: u32 = 0u; j < W; j++) {
    maxVal = max(maxVal, scores[base + j]);
  }

  var expSum: f32 = 0.0;
  for (var j: u32 = 0u; j < W; j++) {
    let e = exp(scores[base + j] - maxVal);
    scores[base + j] = e;
    expSum += e;
  }

  for (var j: u32 = 0u; j < W; j++) {
    scores[base + j] /= expSum;
  }
}
`,ma=`
struct Params {
  n_heads: u32,
  head_dim: u32,
  seq_len: u32,
  window: u32,
}

@group(0) @binding(0) var<storage, read> scores: array<f32>;   // [n_heads, seq_len, window]
@group(0) @binding(1) var<storage, read> v: array<f32>;        // [seq_len, n_heads, head_dim]
@group(0) @binding(2) var<storage, read_write> output: array<f32>; // [seq_len, n_heads, head_dim]
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  // 2D dispatch: gid.x = within (n_heads * head_dim), gid.y = qi
  let hd_idx = gid.x;
  let qi = gid.y;
  let nhd = params.n_heads * params.head_dim;
  if (hd_idx >= nhd || qi >= params.seq_len) { return; }

  let h = hd_idx / params.head_dim;
  let d = hd_idx % params.head_dim;
  let W = params.window;

  // ki_start = max(0, qi - W + 1)
  var ki_start: u32 = 0u;
  if (qi + 1u > W) { ki_start = qi - W + 1u; }

  var sum: f32 = 0.0;
  let scoreBase = h * params.seq_len * W + qi * W;
  for (var slot: u32 = 0u; slot < W; slot++) {
    let ki = ki_start + slot;
    if (ki > qi) { break; }
    let score = scores[scoreBase + slot];
    let vIdx = ki * params.n_heads * params.head_dim + h * params.head_dim + d;
    sum += score * v[vIdx];
  }

  output[qi * nhd + hd_idx] = sum;
}
`,fa=`
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
`,pa=`
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
`,_a=`
struct Params {
  total: u32,    // T * hidden_dim
  grid_x: u32,   // number of workgroups in x dimension (for 2D dispatch)
}

@group(0) @binding(0) var<storage, read_write> gate: array<f32>;  // in-place output
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * params.grid_x * 256u + gid.x;
  if (i >= params.total) { return; }
  let g = gate[i];
  let silu = g / (1.0 + exp(-g));
  gate[i] = silu * up[i];
}
`,la=`
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
`,ha=`
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
`,ga=`
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
`,ba=`
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
`,va=`
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
`,ka=`
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
`,wa=`
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
`;function _(x,s){return Math.ceil(x/s)}function Ea(x,s,i){let t=x.length,a=new Float32Array(t);for(let c=0;c<t;c++)a[c]=x[c]/i;let n=-1/0;for(let c=0;c<t;c++)a[c]>n&&(n=a[c]);let r=0;for(let c=0;c<t;c++)a[c]=Math.exp(a[c]-n),r+=a[c];for(let c=0;c<t;c++)a[c]/=r;let o=Array.from({length:t},(c,d)=>d);o.sort((c,d)=>a[d]-a[c]);let e=0,g=t;for(let c=0;c<t;c++)if(e+=a[o[c]],e>=s){g=c+1;break}let u=0;for(let c=0;c<g;c++)u+=a[o[c]];let q=Math.random()*u,P=0;for(let c=0;c<g;c++)if(P+=a[o[c]],P>=q)return o[c];return o[0]}var oe=class{device=null;config;maxSeqLen;modelBuffers=null;workBuffers=null;pipelines=null;kvCaches=[];position=0;constructor(s={}){this.config=s.config||xe,this.maxSeqLen=s.maxSeqLen||4096}async init(){if(!navigator.gpu)throw new Error("WebGPU not supported");let s=await navigator.gpu.requestAdapter({powerPreference:"high-performance"});if(!s)throw new Error("No WebGPU adapter");let i=[];s.features.has("shader-f16")&&i.push("shader-f16");let t=2*1024*1024*1024,a=s.limits.maxBufferSize,n=s.limits.maxStorageBufferBindingSize,r=a>=t&&n>=t;this.device=await s.requestDevice({requiredFeatures:i,requiredLimits:{maxBufferSize:r?t:a,maxStorageBufferBindingSize:r?t:n}}),this.createWorkBuffers(),this.createKVCaches(),this.createPipelines()}createPipeline(s,i){let t=this.device,a=t.createShaderModule({code:s,label:i});return t.createComputePipeline({layout:"auto",compute:{module:a,entryPoint:"main"},label:i})}createPipelines(){let s=(i,t)=>this.createPipeline(i,t);this.pipelines={matvecF16:s(Ce,"matvecF16"),matvecF16Chunked:s(Ge,"matvecF16Chunked"),matvecF16Offset:s(ra,"matvecF16Offset"),rmsNorm:s(Se,"rmsNorm"),rmsNormOffset:s(sa,"rmsNormOffset"),embeddingLookup:s(Fe,"embeddingLookup"),rope:s(Ae,"rope"),ropeOffset:s($e,"ropeOffset"),attnScore:s(Te,"attnScore"),softmax:s(ze,"softmax"),attnValue:s(Me,"attnValue"),kvCacheWrite:s(Oe,"kvCacheWrite"),swiGLU:s(We,"swiGLU"),addVectors:s(Ee,"addVectors"),addVectorsOffset:s(oa,"addVectorsOffset"),addInPlace:s(Le,"addInPlace"),addInPlaceOffset:s(ia,"addInPlaceOffset"),copyBuffer:s(De,"copyBuffer"),copyBufferOffset:s(da,"copyBufferOffset"),timeEmbedding:s(Ie,"timeEmbedding"),eulerStep:s(Re,"eulerStep"),cfgCombine:s(Ne,"cfgCombine"),fsqQuantize:s(Ve,"fsqQuantize"),biAttnScore:s(je,"biAttnScore"),biSoftmax:s(Ke,"biSoftmax"),biAttnValue:s(He,"biAttnValue"),swiGLUOffset:s(na,"swiGLUOffset"),zeroFill:s(wa,"zeroFill"),multiCodebookEmbed:s(ka,"multiCodebookEmbed"),vqLookup:s(Ye,"vqLookup"),fsqDequant:s(Xe,"fsqDequant"),causalConv1d:s(Ze,"causalConv1d"),causalConvTranspose1d:s(ea,"causalConvTranspose1d"),convTransposeNormScale:s(Je,"convTransposeNormScale"),layerScale:s(aa,"layerScale"),alibiAttnScore:s(ca,"alibiAttnScore"),codecSoftmax:s(ua,"codecSoftmax"),codecAttnValue:s(ma,"codecAttnValue"),batchedMatvecF16:s(fa,"batchedMatvecF16"),batchedRmsNorm:s(pa,"batchedRmsNorm"),batchedSwiGLU:s(_a,"batchedSwiGLU"),batchedAdd:s(la,"batchedAdd"),batchedCopy:s(ha,"batchedCopy"),batchedLayerScale:s(ga,"batchedLayerScale"),qkNorm:s(ba,"qkNorm"),concatCodecInput:s(va,"concatCodecInput"),argmax:s(ta,"argmax"),normalizeCodebook:s(Qe,"normalizeCodebook")}}createUniform(s){let i=this.device,t=Math.ceil(s.byteLength/16)*16,a=i.createBuffer({size:t,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,mappedAtCreation:!0});return new Uint8Array(a.getMappedRange()).set(new Uint8Array(s)),a.unmap(),a}packUniform(s){let i=new ArrayBuffer(s.length*4),t=new Uint32Array(i),a=new Float32Array(i);for(let n=0;n<s.length;n++){let r=s[n];r.u!==void 0?t[n]=r.u:r.f!==void 0&&(a[n]=r.f)}return this.createUniform(i)}createWorkBuffers(){let s=this.device,i=this.config.backbone,t=this.config.fm,a=GPUBufferUsage.STORAGE,n=GPUBufferUsage.COPY_SRC,r=GPUBufferUsage.COPY_DST,o=(e,g,u=0)=>s.createBuffer({size:e,usage:a|n|r|u,label:g});this.workBuffers={hidden:o(i.dim*4,"hidden"),residual:o(i.dim*4,"residual"),normed:o(i.dim*4,"normed"),q:o(i.n_heads*i.head_dim*4,"q"),k:o(i.n_kv_heads*i.head_dim*4,"k"),v:o(i.n_kv_heads*i.head_dim*4,"v"),attn_out:o(i.n_heads*i.head_dim*4,"attn_out"),scores:o(i.n_heads*this.maxSeqLen*4,"scores"),gate:o(i.hidden_dim*4,"gate"),up:o(i.hidden_dim*4,"up"),down:o(i.dim*4,"down"),x_t:o(t.n_acoustic_out*4,"x_t"),velocity:o(t.n_acoustic_out*4,"velocity"),v_uncond:o(t.n_acoustic_out*4,"v_uncond"),time_embed:o(t.dim*4,"time_embed"),time_proj:o(t.dim*4,"time_proj"),x_t_proj:o(t.dim*4,"x_t_proj"),fm_hidden:o(t.dim*4,"fm_hidden"),fm_residual:o(t.dim*4,"fm_residual"),fm_normed:o(t.dim*4,"fm_normed"),fm_q:o(3*t.n_heads*t.head_dim*4,"fm_q"),fm_k:o(3*t.n_kv_heads*t.head_dim*4,"fm_k"),fm_v:o(3*t.n_kv_heads*t.head_dim*4,"fm_v"),fm_attn_out:o(3*t.n_heads*t.head_dim*4,"fm_attn_out"),fm_scores:o(t.n_heads*3*3*4,"fm_scores"),fm_seq:o(3*t.dim*4,"fm_seq"),fm_gate:o(3*t.hidden_dim*4,"fm_gate"),fm_up:o(3*t.hidden_dim*4,"fm_up"),fm_down:o(3*t.dim*4,"fm_down"),semantic_logits:o(t.semantic_vocab*4,"semantic_logits"),semantic_argmax:o(4,"semantic_argmax"),acoustic_out:o(t.n_acoustic_out*4,"acoustic_out"),acoustic_codes:o(t.n_acoustic_out*4,"acoustic_codes"),logits:o(i.vocab_size*4,"logits"),argmax_result:o(4,"argmax_result")}}createKVCaches(){let s=this.device,i=this.config.backbone,t=i.n_kv_heads*i.head_dim;this.kvCaches=[];for(let a=0;a<i.n_layers;a++)this.kvCaches.push({k:s.createBuffer({size:this.maxSeqLen*t*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,label:`kv_cache.${a}.k`}),v:s.createBuffer({size:this.maxSeqLen*t*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,label:`kv_cache.${a}.v`})})}async loadWeights(s,i){let t=this.device,a=await Ue(s),n=i||(()=>{});n({loaded:0,total:3,component:"all",tensor:"Loading backbone..."});let r=await ue(t,s,a,"backbone",i);n({loaded:1,total:3,component:"all",tensor:"Loading FM transformer..."});let o=await ue(t,s,a,"fm",i);n({loaded:2,total:3,component:"all",tensor:"Loading codec decoder..."});let e=await ue(t,s,a,"codec",i);this.modelBuffers=this.organizeWeights(r,o,e),n({loaded:3,total:3,component:"all",tensor:"Done!"})}async loadWeightsFromHF(s=re,i){let t=this.device,{backbone:a,fm:n,codec:r}=await qe(t,s,i);this.modelBuffers=this.organizeWeights(a,n,r),await this.normalizeVQCodebook(),await this.precomputeConvTransposeScales()}async normalizeVQCodebook(){let s=this.device,i=this.pipelines,t=this.modelBuffers,a=this.config.codec,n=this.packUniform([{u:a.semantic_codebook_size},{u:a.semantic_dim},{f:1e-5}]),r=s.createCommandEncoder({label:"normalize_codebook"}),o=r.beginComputePass({label:"normalize_codebook"});this.dispatch(o,i.normalizeCodebook,[t.codec_semantic_codebook,t.codec_cluster_usage,n],[_(a.semantic_codebook_size*a.semantic_dim/2,128)]),o.end(),s.queue.submit([r.finish()]),await s.queue.onSubmittedWorkDone()}async precomputeConvTransposeScales(){let s=this.device,i=this.pipelines,t=this.modelBuffers,a=this.config.codec,n=s.createCommandEncoder({label:"precompute_conv_transpose_scales"});for(let r=0;r<a.decoder_stages;r++){let o=t.codec_stages[r];if(!o.conv_w||!o.conv_g)continue;let e=s.createBuffer({size:a.dim*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,label:`codec_conv_transpose_scale_s${r}`});o.conv_scale=e;let u=this.packUniform([{u:a.dim},{u:a.dim},{u:4}]),q=n.beginComputePass({label:`conv_transpose_norm_scale_s${r}`});this.dispatch(q,i.convTransposeNormScale,[o.conv_w,o.conv_g,e,u],[a.dim]),q.end()}s.queue.submit([n.finish()]),await s.queue.onSubmittedWorkDone()}organizeWeights(s,i,t){let a=(e,g)=>{let u=e.buffers.get(g);if(!u)throw new Error(`Missing weight: ${g}`);return u},n=[];for(let e=0;e<this.config.backbone.n_layers;e++)n.push({attn_norm:a(s,`layers.${e}.attention_norm.weight`),wq:a(s,`layers.${e}.attention.wq.weight`),wk:a(s,`layers.${e}.attention.wk.weight`),wv:a(s,`layers.${e}.attention.wv.weight`),wo:a(s,`layers.${e}.attention.wo.weight`),ffn_norm:a(s,`layers.${e}.ffn_norm.weight`),w1:a(s,`layers.${e}.feed_forward.w1.weight`),w2:a(s,`layers.${e}.feed_forward.w2.weight`),w3:a(s,`layers.${e}.feed_forward.w3.weight`)});let r=[];for(let e=0;e<this.config.fm.n_layers;e++)r.push({attn_norm:a(i,`acoustic_transformer.layers.${e}.attention_norm.weight`),wq:a(i,`acoustic_transformer.layers.${e}.attention.wq.weight`),wk:a(i,`acoustic_transformer.layers.${e}.attention.wk.weight`),wv:a(i,`acoustic_transformer.layers.${e}.attention.wv.weight`),wo:a(i,`acoustic_transformer.layers.${e}.attention.wo.weight`),ffn_norm:a(i,`acoustic_transformer.layers.${e}.ffn_norm.weight`),w1:a(i,`acoustic_transformer.layers.${e}.feed_forward.w1.weight`),w2:a(i,`acoustic_transformer.layers.${e}.feed_forward.w2.weight`),w3:a(i,`acoustic_transformer.layers.${e}.feed_forward.w3.weight`)});let o=[];for(let e=0;e<4;e++){let g=1+e*2,u=2+e*2,q=e<3;o.push({transformer_layers:this.getCodecTransformerLayers(t,g),...q?{conv_w:a(t,`audio_tokenizer.decoder_blocks.${u}.conv.parametrizations.weight.original1`),conv_g:a(t,`audio_tokenizer.decoder_blocks.${u}.conv.parametrizations.weight.original0`)}:{}})}return{tok_embeddings:a(s,"mm_audio_embeddings.tok_embeddings.weight"),audio_embeddings:a(s,"mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight"),backbone_layers:n,final_norm:a(s,"norm.weight"),fm_input_proj:a(i,"acoustic_transformer.input_projection.weight"),fm_llm_proj:a(i,"acoustic_transformer.llm_projection.weight"),fm_time_proj:a(i,"acoustic_transformer.time_projection.weight"),fm_layers:r,fm_norm:a(i,"acoustic_transformer.norm.weight"),fm_semantic_out:a(i,"acoustic_transformer.semantic_codebook_output.weight"),fm_acoustic_out:a(i,"acoustic_transformer.acoustic_codebook_output.weight"),codec_input_conv_w:a(t,"audio_tokenizer.decoder_blocks.0.conv.parametrizations.weight.original1"),codec_input_conv_g:a(t,"audio_tokenizer.decoder_blocks.0.conv.parametrizations.weight.original0"),codec_stages:o,codec_output_conv_w:a(t,"audio_tokenizer.output_proj.conv.parametrizations.weight.original1"),codec_output_conv_g:a(t,"audio_tokenizer.output_proj.conv.parametrizations.weight.original0"),codec_semantic_codebook:a(t,"audio_tokenizer.quantizer.semantic_codebook.embedding_sum"),codec_cluster_usage:a(t,"audio_tokenizer.quantizer.semantic_codebook.cluster_usage")}}getCodecTransformerLayers(s,i){let t=n=>{let r=s.buffers.get(n);if(!r)throw new Error(`Missing codec weight: ${n}`);return r},a=[];for(let n=0;n<2;n++){let r=`audio_tokenizer.decoder_blocks.${i}.layers.${n}`;a.push({attn_norm:t(`${r}.attention_norm.weight`),q_norm:t(`${r}.attention.q_norm.weight`),k_norm:t(`${r}.attention.k_norm.weight`),wq:t(`${r}.attention.wq.weight`),wk:t(`${r}.attention.wk.weight`),wv:t(`${r}.attention.wv.weight`),wo:t(`${r}.attention.wo.weight`),attn_scale:t(`${r}.attention_scale`),ffn_norm:t(`${r}.ffn_norm.weight`),w1:t(`${r}.feed_forward.w1.weight`),w2:t(`${r}.feed_forward.w2.weight`),w3:t(`${r}.feed_forward.w3.weight`),ffn_scale:t(`${r}.ffn_scale`)})}return a}dispatch(s,i,t,a){let n=t.map((o,e)=>({binding:e,resource:{buffer:o}})),r=this.device.createBindGroup({layout:i.getBindGroupLayout(0),entries:n});s.setPipeline(i),s.setBindGroup(0,r),s.dispatchWorkgroups(...a)}backboneStep(s,i,t=!1,a){let n=this.pipelines,r=this.workBuffers,o=this.modelBuffers,e=this.config.backbone,g=this.position,u;u=s.beginComputePass({label:`embed_pos${g}`});let q=this.packUniform([{u:i},{u:e.dim}]),P=t?o.audio_embeddings:o.tok_embeddings;if(this.dispatch(u,n.embeddingLookup,[P,r.hidden,q],[_(e.dim,256)]),u.end(),a){u=s.beginComputePass({label:`voice_embed_pos${g}`});let f=this.packUniform([{u:e.dim}]);this.dispatch(u,n.copyBuffer,[a,r.hidden,f],[_(e.dim,256)]),u.end()}for(let f=0;f<e.n_layers;f++){let b=o.backbone_layers[f],G=this.kvCaches[f];u=s.beginComputePass({label:`layer${f}_attn`});let C=this.packUniform([{u:e.dim}]);this.dispatch(u,n.copyBuffer,[r.hidden,r.residual,C],[_(e.dim,256)]);let m=this.packUniform([{u:e.dim},{f:e.norm_eps}]);this.dispatch(u,n.rmsNorm,[r.hidden,b.attn_norm,r.normed,m],[1]),u.end(),u=s.beginComputePass({label:`layer${f}_qkv`});let h=this.packUniform([{u:e.n_heads*e.head_dim},{u:e.dim}]);this.dispatch(u,n.matvecF16,[b.wq,r.normed,r.q,h],[e.n_heads*e.head_dim]);let l=this.packUniform([{u:e.n_kv_heads*e.head_dim},{u:e.dim}]);this.dispatch(u,n.matvecF16,[b.wk,r.normed,r.k,l],[e.n_kv_heads*e.head_dim]),this.dispatch(u,n.matvecF16,[b.wv,r.normed,r.v,l],[e.n_kv_heads*e.head_dim]),u.end(),u=s.beginComputePass({label:`layer${f}_rope_attn`});let w=this.packUniform([{u:e.head_dim},{u:g},{u:e.n_heads},{f:e.rope_theta}]);this.dispatch(u,n.rope,[r.q,w],[_(e.n_heads*e.head_dim/2,64)]);let p=this.packUniform([{u:e.head_dim},{u:g},{u:e.n_kv_heads},{f:e.rope_theta}]);this.dispatch(u,n.rope,[r.k,p],[_(e.n_kv_heads*e.head_dim/2,64)]);let B=this.packUniform([{u:g},{u:e.n_kv_heads*e.head_dim}]);this.dispatch(u,n.kvCacheWrite,[r.k,r.v,G.k,G.v,B],[_(e.n_kv_heads*e.head_dim,256)]);let k=g+1,v=e.n_heads/e.n_kv_heads,S=this.packUniform([{u:e.n_heads},{u:e.n_kv_heads},{u:e.head_dim},{u:k},{u:v}]);this.dispatch(u,n.attnScore,[r.q,G.k,r.scores,S],[_(e.n_heads*k,64)]),u.end(),u=s.beginComputePass({label:`layer${f}_attn_out`});let F=this.packUniform([{u:e.n_heads},{u:k}]);this.dispatch(u,n.softmax,[r.scores,F],[e.n_heads]);let T=this.packUniform([{u:e.n_heads},{u:e.n_kv_heads},{u:e.head_dim},{u:k},{u:v}]);this.dispatch(u,n.attnValue,[r.scores,G.v,r.attn_out,T],[_(e.n_heads*e.head_dim,128)]),u.end(),u=s.beginComputePass({label:`layer${f}_wo_res`});let M=this.packUniform([{u:e.dim},{u:e.n_heads*e.head_dim}]);this.dispatch(u,n.matvecF16,[b.wo,r.attn_out,r.hidden,M],[e.dim]),u.end(),u=s.beginComputePass({label:`layer${f}_res1`});let O=this.packUniform([{u:e.dim}]);this.dispatch(u,n.addInPlace,[r.hidden,r.residual,O],[_(e.dim,256)]),this.dispatch(u,n.copyBuffer,[r.hidden,r.residual,C],[_(e.dim,256)]);let z=this.packUniform([{u:e.dim},{f:e.norm_eps}]);this.dispatch(u,n.rmsNorm,[r.hidden,b.ffn_norm,r.normed,z],[1]),u.end(),u=s.beginComputePass({label:`layer${f}_ffn`});let E=this.packUniform([{u:e.hidden_dim},{u:e.dim}]);this.dispatch(u,n.matvecF16,[b.w1,r.normed,r.gate,E],[e.hidden_dim]),this.dispatch(u,n.matvecF16,[b.w3,r.normed,r.up,E],[e.hidden_dim]),u.end(),u=s.beginComputePass({label:`layer${f}_ffn_out`});let L=this.packUniform([{u:e.hidden_dim}]);this.dispatch(u,n.swiGLU,[r.gate,r.up,L],[_(e.hidden_dim,256)]);let N=this.packUniform([{u:e.dim},{u:e.hidden_dim}]);this.dispatch(u,n.matvecF16,[b.w2,r.gate,r.hidden,N],[e.dim]),u.end(),u=s.beginComputePass({label:`layer${f}_res2`}),this.dispatch(u,n.addInPlace,[r.hidden,r.residual,O],[_(e.dim,256)]),u.end()}u=s.beginComputePass({label:"final_norm"});let c=this.packUniform([{u:e.dim},{f:e.norm_eps}]);this.dispatch(u,n.rmsNorm,[r.hidden,o.final_norm,r.normed,c],[1]),u.end(),u=s.beginComputePass({label:"lm_head"});for(let b=0;b<e.vocab_size;b+=65535){let G=Math.min(65535,e.vocab_size-b),C=this.packUniform([{u:G},{u:e.dim},{u:b}]);this.dispatch(u,n.matvecF16Chunked,[o.tok_embeddings,r.normed,r.logits,C],[G])}u.end(),u=s.beginComputePass({label:"argmax"});let d=this.packUniform([{u:e.vocab_size}]);this.dispatch(u,n.argmax,[r.logits,r.argmax_result,d],[1]),u.end()}fmTransformerPass(s,i){let t=this.pipelines,a=this.workBuffers,n=this.modelBuffers,r=this.config.fm,o=r.dim,e=3,g=r.n_heads*r.head_dim,u=r.n_kv_heads*r.head_dim,q=r.n_heads/r.n_kv_heads;for(let P=0;P<r.n_layers;P++){let c=n.fm_layers[P],d;d=s.beginComputePass({label:`fm_l${P}_attn_prep`});let f=this.packUniform([{u:e*o},{u:0},{u:0}]);this.dispatch(d,t.copyBufferOffset,[a.fm_seq,a.fm_down,f],[_(e*o,256)]);for(let m=0;m<e;m++){let h=m*o,l=this.packUniform([{u:o},{f:1e-5},{u:h},{u:h}]);this.dispatch(d,t.rmsNormOffset,[a.fm_seq,c.attn_norm,a.fm_gate,l],[1])}d.end(),d=s.beginComputePass({label:`fm_l${P}_qkv`});for(let m=0;m<e;m++){let h=m*o,l=m*g,w=m*u,p=this.packUniform([{u:g},{u:o},{u:h},{u:l}]);this.dispatch(d,t.matvecF16Offset,[c.wq,a.fm_gate,a.fm_q,p],[g]);let B=this.packUniform([{u},{u:o},{u:h},{u:w}]);this.dispatch(d,t.matvecF16Offset,[c.wk,a.fm_gate,a.fm_k,B],[u]),this.dispatch(d,t.matvecF16Offset,[c.wv,a.fm_gate,a.fm_v,B],[u])}d.end(),d=s.beginComputePass({label:`fm_l${P}_attn`});let b=this.packUniform([{u:r.n_heads},{u:r.n_kv_heads},{u:r.head_dim},{u:e},{u:q}]);this.dispatch(d,t.biAttnScore,[a.fm_q,a.fm_k,a.fm_scores,b],[_(r.n_heads*e*e,64)]),d.end(),d=s.beginComputePass({label:`fm_l${P}_attn_val`});let G=this.packUniform([{u:r.n_heads},{u:e}]);this.dispatch(d,t.biSoftmax,[a.fm_scores,G],[_(r.n_heads*e,64)]);let C=this.packUniform([{u:r.n_heads},{u:r.n_kv_heads},{u:r.head_dim},{u:e},{u:q}]);this.dispatch(d,t.biAttnValue,[a.fm_scores,a.fm_v,a.fm_attn_out,C],[_(e*r.n_heads*r.head_dim,64)]),d.end(),d=s.beginComputePass({label:`fm_l${P}_wo_res`});for(let m=0;m<e;m++){let h=m*g,l=m*o,w=this.packUniform([{u:o},{u:g},{u:h},{u:l}]);this.dispatch(d,t.matvecF16Offset,[c.wo,a.fm_attn_out,a.fm_seq,w],[o])}d.end(),d=s.beginComputePass({label:`fm_l${P}_res1`});for(let m=0;m<e;m++){let h=m*o,l=this.packUniform([{u:o},{u:h},{u:h}]);this.dispatch(d,t.addInPlaceOffset,[a.fm_seq,a.fm_down,l],[_(o,256)])}this.dispatch(d,t.copyBufferOffset,[a.fm_seq,a.fm_down,f],[_(e*o,256)]),d.end(),d=s.beginComputePass({label:`fm_l${P}_ffn`});for(let m=0;m<e;m++){let h=m*o,l=m*r.hidden_dim,w=this.packUniform([{u:o},{f:1e-5},{u:h},{u:0}]);this.dispatch(d,t.rmsNormOffset,[a.fm_seq,c.ffn_norm,a.fm_normed,w],[1]);let p=this.packUniform([{u:r.hidden_dim},{u:o},{u:0},{u:l}]);this.dispatch(d,t.matvecF16Offset,[c.w1,a.fm_normed,a.fm_gate,p],[r.hidden_dim]),this.dispatch(d,t.matvecF16Offset,[c.w3,a.fm_normed,a.fm_up,p],[r.hidden_dim])}d.end(),d=s.beginComputePass({label:`fm_l${P}_ffn_act`});for(let m=0;m<e;m++){let h=m*r.hidden_dim,l=this.packUniform([{u:r.hidden_dim},{u:h},{u:h}]);this.dispatch(d,t.swiGLUOffset,[a.fm_gate,a.fm_up,l],[_(r.hidden_dim,256)])}d.end(),d=s.beginComputePass({label:`fm_l${P}_ffn_down`});for(let m=0;m<e;m++){let h=m*r.hidden_dim,l=m*o,w=this.packUniform([{u:o},{u:r.hidden_dim},{u:h},{u:l}]);this.dispatch(d,t.matvecF16Offset,[c.w2,a.fm_gate,a.fm_seq,w],[o])}d.end(),d=s.beginComputePass({label:`fm_l${P}_res2`});for(let m=0;m<e;m++){let h=m*o,l=this.packUniform([{u:o},{u:h},{u:h}]);this.dispatch(d,t.addInPlaceOffset,[a.fm_seq,a.fm_down,l],[_(o,256)])}d.end()}{let P=s.beginComputePass({label:"fm_final_norm_vel"}),c=this.packUniform([{u:o},{f:1e-5},{u:0},{u:0}]);this.dispatch(P,t.rmsNormOffset,[a.fm_seq,n.fm_norm,a.fm_normed,c],[1]);let d=this.packUniform([{u:r.n_acoustic_out},{u:o}]);this.dispatch(P,t.matvecF16,[n.fm_acoustic_out,a.fm_normed,i,d],[r.n_acoustic_out]),P.end()}}fmForward(s,i){let t=this.pipelines,a=this.workBuffers,n=this.modelBuffers,r=this.config.fm,o=r.dim,e;e=s.beginComputePass({label:"fm_init"});let g=this.packUniform([{u:r.semantic_vocab},{u:o}]);this.dispatch(e,t.matvecF16,[n.fm_semantic_out,a.normed,a.semantic_logits,g],[r.semantic_vocab]);let u=this.packUniform([{u:o},{u:o}]);this.dispatch(e,t.matvecF16,[n.fm_llm_proj,a.normed,a.fm_hidden,u],[o]);{let c=i??new Float32Array(r.n_acoustic_out);if(!i)for(let d=0;d<r.n_acoustic_out;d++){let f=Math.random(),b=Math.random();c[d]=Math.sqrt(-2*Math.log(f))*Math.cos(2*Math.PI*b)}this.device.queue.writeBuffer(a.x_t,0,c)}e.end(),e=s.beginComputePass({label:"fm_semantic_argmax"});let q=this.packUniform([{u:r.semantic_vocab}]);this.dispatch(e,t.argmax,[a.semantic_logits,a.semantic_argmax,q],[1]),e.end();for(let c=0;c<r.nfe-1;c++){let d=c/(r.nfe-1),f=1/(r.nfe-1);e=s.beginComputePass({label:`fm_step${c}_prep`});let b=this.packUniform([{u:o},{f:d}]);this.dispatch(e,t.timeEmbedding,[a.time_embed,b],[_(o/2,256)]),e.end(),e=s.beginComputePass({label:`fm_step${c}_proj`});let G=this.packUniform([{u:o},{u:o}]);this.dispatch(e,t.matvecF16,[n.fm_time_proj,a.time_embed,a.time_proj,G],[o]);let C=this.packUniform([{u:o},{u:r.n_acoustic_out}]);this.dispatch(e,t.matvecF16,[n.fm_input_proj,a.x_t,a.x_t_proj,C],[o]),e.end(),e=s.beginComputePass({label:`fm_step${c}_assemble`});let m=this.packUniform([{u:o},{u:0},{u:0}]);this.dispatch(e,t.copyBufferOffset,[a.x_t_proj,a.fm_seq,m],[_(o,256)]);let h=this.packUniform([{u:o},{u:0},{u:o}]);this.dispatch(e,t.copyBufferOffset,[a.time_proj,a.fm_seq,h],[_(o,256)]);let l=this.packUniform([{u:o},{u:0},{u:2*o}]);this.dispatch(e,t.copyBufferOffset,[a.fm_hidden,a.fm_seq,l],[_(o,256)]),e.end(),this.fmTransformerPass(s,a.velocity),e=s.beginComputePass({label:`fm_step${c}_uncond`}),this.dispatch(e,t.copyBufferOffset,[a.x_t_proj,a.fm_seq,m],[_(o,256)]),this.dispatch(e,t.copyBufferOffset,[a.time_proj,a.fm_seq,h],[_(o,256)]);let w=this.packUniform([{u:o}]);this.dispatch(e,t.zeroFill,[a.fm_residual,w],[_(o,256)]),this.dispatch(e,t.copyBufferOffset,[a.fm_residual,a.fm_seq,l],[_(o,256)]),e.end(),this.fmTransformerPass(s,a.v_uncond),e=s.beginComputePass({label:`fm_step${c}_euler`});let p=this.packUniform([{u:r.n_acoustic_out},{f:r.cfg_alpha}]);this.dispatch(e,t.cfgCombine,[a.velocity,a.v_uncond,p],[_(r.n_acoustic_out,64)]);let B=this.packUniform([{u:r.n_acoustic_out},{f}]);this.dispatch(e,t.eulerStep,[a.x_t,a.velocity,B],[_(r.n_acoustic_out,64)]),e.end()}e=s.beginComputePass({label:"fm_fsq"});let P=this.packUniform([{u:r.n_acoustic_out},{u:this.config.codec.acoustic_codebook_size},{u:2}]);this.dispatch(e,t.fsqQuantize,[a.x_t,a.acoustic_codes,P],[_(r.n_acoustic_out,64)]),e.end()}async codecDecode(s,i){let t=this.device,a=this.pipelines,n=this.modelBuffers,r=this.config.codec,o=s.length,e=r.dim,g=this.uploadArray(s),u=this.uploadArray(i),q=this.createGPUBuffer(o*r.semantic_dim*4,"codec_sem_embed"),P=this.createGPUBuffer(o*r.n_acoustic_codebook*4,"codec_ac_float"),c=this.createGPUBuffer(o*292*4,"codec_concat"),d=o,f=this.createGPUBuffer(d*e*4,"codec_cur"),b=this.createGPUBuffer(d*e*4,"codec_tmp"),G=t.createCommandEncoder({label:"codec_decode_pre"}),C=[],m=(y,$,U,A)=>{let W=G.beginComputePass({label:A});this.dispatch(W,y,$,U),W.end()},h=async y=>{t.pushErrorScope("validation"),t.queue.submit([G.finish()]),await t.queue.onSubmittedWorkDone();let $=await t.popErrorScope();$&&(globalThis.__codecError=`${y}: ${$.message}`);for(let U of C)U.destroy();C=[]},l=this.packUniform([{u:o},{u:r.semantic_dim}]);m(a.vqLookup,[g,n.codec_semantic_codebook,q,l],[_(o*r.semantic_dim,128)],"codec_vq");let w=this.packUniform([{u:o},{u:r.n_acoustic_codebook},{u:r.acoustic_codebook_size},{u:2}]);m(a.fsqDequant,[u,P,w],[_(o*r.n_acoustic_codebook,64)],"codec_fsq");let p=this.packUniform([{u:o},{u:r.semantic_dim},{u:r.n_acoustic_codebook}]);m(a.concatCodecInput,[q,P,c,p],[_(o*292,256)],"codec_concat");let B=this.packUniform([{u:292},{u:e},{u:3},{u:d},{u:1}]);m(a.causalConv1d,[c,n.codec_input_conv_w,n.codec_input_conv_g,f,B],[_(d,64),e],"codec_input_conv"),C.push(g,u,q,P,c),await h("codec_preprocess");let k=new Map,v=(y,$)=>{let U=k.get(y);return U&&U.length>0?U.pop():this.createGPUBuffer(y,$)},S=(...y)=>{for(let $ of y){let U=$.size,A=k.get(U);A||(A=[],k.set(U,A)),A.push($)}},F=()=>{for(let y of k.values())for(let $ of y)$.destroy();k.clear()},T=[2,2,2,1],M=[4,4,4,3],O=[2,4,8,16];for(let y=0;y<r.decoder_stages;y++){let $=n.codec_stages[y];for(let U=0;U<r.decoder_layers_per_stage;U++){G=t.createCommandEncoder({label:`codec_s${y}_l${U}`});let A=$.transformer_layers[U],W=d*e*4;b.size<W&&(b.destroy(),b=this.createGPUBuffer(W,"codec_tmp"));let H=b,Y=d*e,ae=this.packUniform([{u:Y}]);m(a.batchedCopy,[f,H,ae],[_(Y,256)],`codec_s${y}_l${U}_copy_res`);let se=this.packUniform([{u:e},{f:r.norm_eps},{u:d}]),I=v(W,"codec_attn_normed");m(a.batchedRmsNorm,[f,A.attn_norm,I,se],[d],`codec_s${y}_l${U}_attn_norm`);let R=v(W,"codec_q"),V=v(W,"codec_k"),Z=v(W,"codec_v"),J=this.packUniform([{u:e},{u:e},{u:d}]);m(a.batchedMatvecF16,[A.wq,I,R,J],[e,d],`codec_s${y}_l${U}_qproj`),m(a.batchedMatvecF16,[A.wk,I,V,J],[e,d],`codec_s${y}_l${U}_kproj`),m(a.batchedMatvecF16,[A.wv,I,Z,J],[e,d],`codec_s${y}_l${U}_vproj`);let te=this.packUniform([{u:r.n_heads},{u:r.head_dim},{u:d},{f:r.qk_norm_eps}]);m(a.qkNorm,[R,A.q_norm,te],[_(d*r.n_heads,128)],`codec_s${y}_l${U}_qnorm`),m(a.qkNorm,[V,A.k_norm,te],[_(d*r.n_heads,128)],`codec_s${y}_l${U}_knorm`);let ee=O[y],me=r.n_heads*d*ee*4,Q=v(me,"codec_scores"),fe=this.packUniform([{u:r.n_heads},{u:r.head_dim},{u:d},{u:ee}]);m(a.alibiAttnScore,[R,V,Q,fe],[_(ee,64),d,r.n_heads],`codec_s${y}_l${U}_attn_score`);let Pa=this.packUniform([{u:r.n_heads},{u:d},{u:ee}]);m(a.codecSoftmax,[Q,Pa],[_(r.n_heads*d,64)],`codec_s${y}_l${U}_softmax`);let pe=v(W,"codec_attn_out"),ya=this.packUniform([{u:r.n_heads},{u:r.head_dim},{u:d},{u:ee}]);m(a.codecAttnValue,[Q,Z,pe,ya],[_(r.n_heads*r.head_dim,64),d],`codec_s${y}_l${U}_attn_val`);let _e=v(W,"codec_wo_out");m(a.batchedMatvecF16,[A.wo,pe,_e,J],[e,d],`codec_s${y}_l${U}_wo`);let ke=this.packUniform([{u:e},{u:Y}]);m(a.batchedLayerScale,[_e,A.attn_scale,H,f,ke],[_(Y,256)],`codec_s${y}_l${U}_attn_res`),m(a.batchedCopy,[f,H,ae],[_(Y,256)],`codec_s${y}_l${U}_copy_ffn_res`);let ne=v(W,"codec_ffn_normed");m(a.batchedRmsNorm,[f,A.ffn_norm,ne,se],[d],`codec_s${y}_l${U}_ffn_norm`);let de=d*r.hidden_dim,we=de*4,ce=v(we,"codec_gate"),le=v(we,"codec_up"),Pe=this.packUniform([{u:r.hidden_dim},{u:e},{u:d}]);m(a.batchedMatvecF16,[A.w1,ne,ce,Pe],[r.hidden_dim,d],`codec_s${y}_l${U}_gate`),m(a.batchedMatvecF16,[A.w3,ne,le,Pe],[r.hidden_dim,d],`codec_s${y}_l${U}_up`);let he=Math.min(_(de,256),65535),xa=_(_(de,256),he),Ua=this.packUniform([{u:de},{u:he}]);m(a.batchedSwiGLU,[ce,le,Ua],[he,xa],`codec_s${y}_l${U}_swiglu`);let ge=v(W,"codec_down"),Ba=this.packUniform([{u:e},{u:r.hidden_dim},{u:d}]);m(a.batchedMatvecF16,[A.w2,ce,ge,Ba],[e,d],`codec_s${y}_l${U}_down`),m(a.batchedLayerScale,[ge,A.ffn_scale,H,f,ke],[_(Y,256)],`codec_s${y}_l${U}_ffn_res`),t.pushErrorScope("validation"),t.queue.submit([G.finish()]),await t.queue.onSubmittedWorkDone();let ye=await t.popErrorScope();ye&&(globalThis.__codecError=`codec_s${y}_l${U}: ${ye.message}`),S(I,R,V,Z,Q,pe,_e,ne,ce,le,ge)}if($.conv_w&&$.conv_scale&&T[y]>1){F(),G=t.createCommandEncoder({label:`codec_s${y}_conv`});let U=d*T[y],A=this.createGPUBuffer(U*e*4,"codec_upsampled"),W=this.packUniform([{u:e},{u:e},{u:M[y]},{u:U},{u:T[y]}]);m(a.causalConvTranspose1d,[f,$.conv_w,$.conv_scale,A,W],[_(U,64),e],`codec_s${y}_conv_up`),C.push(f),f=A,d=U,await h(`codec_s${y}_conv`)}}F(),G=t.createCommandEncoder({label:"codec_output"});let z=d,E=this.createGPUBuffer(z*r.patch_size*4,"codec_output"),L=this.packUniform([{u:e},{u:r.patch_size},{u:7},{u:z},{u:1}]);m(a.causalConv1d,[f,n.codec_output_conv_w,n.codec_output_conv_g,E,L],[_(z,64),r.patch_size],"codec_output_conv"),C.push(f,b),await h("codec_output");let N=z*r.patch_size,K=await this.readF32Array(E,N),X=0;for(let y=0;y<Math.min(K.length,1e3);y++)K[y]!==0&&X++;return globalThis.__codecDebug={outT:z,patchSize:r.patch_size,totalSamples:N,nonZero:X,first5:Array.from(K.slice(0,5)),curT:d},E.destroy(),K}async debugCodecDecode(s,i){let t=this.device,a=this.pipelines,n=this.modelBuffers,r=this.config.codec,o=s.length,e=r.dim,g={},u=this.uploadArray(s),q=this.uploadArray(i),P=this.createGPUBuffer(o*r.semantic_dim*4,"codec_sem_embed"),c=this.createGPUBuffer(o*r.n_acoustic_codebook*4,"codec_ac_float"),d=this.createGPUBuffer(o*292*4,"codec_concat"),f=o,b=this.createGPUBuffer(f*e*4,"codec_cur"),G=this.createGPUBuffer(f*e*4,"codec_tmp"),C=[],m=(p,B,k,v,S)=>{let F=p.beginComputePass({label:S});this.dispatch(F,B,k,v),F.end()};{let p=t.createCommandEncoder({label:"codec_phase1"}),B=this.packUniform([{u:o},{u:r.semantic_dim}]);m(p,a.vqLookup,[u,n.codec_semantic_codebook,P,B],[_(o*r.semantic_dim,128)],"codec_vq");let k=this.packUniform([{u:o},{u:r.n_acoustic_codebook},{u:r.acoustic_codebook_size},{u:2}]);m(p,a.fsqDequant,[q,c,k],[_(o*r.n_acoustic_codebook,64)],"codec_fsq");let v=this.packUniform([{u:o},{u:r.semantic_dim},{u:r.n_acoustic_codebook}]);m(p,a.concatCodecInput,[P,c,d,v],[_(o*292,256)],"codec_concat");let S=this.packUniform([{u:292},{u:e},{u:3},{u:f},{u:1}]);m(p,a.causalConv1d,[d,n.codec_input_conv_w,n.codec_input_conv_g,b,S],[_(f,64),e],"codec_input_conv"),t.queue.submit([p.finish()]),await t.queue.onSubmittedWorkDone()}g.vq_embed=await this.readF32Array(P,o*r.semantic_dim),g.fsq_dequant=await this.readF32Array(c,o*r.n_acoustic_codebook),g.concat=await this.readF32Array(d,o*292),g.after_input_conv=await this.readF32Array(b,f*e);let h=[2,2,2,1],l=[4,4,4,3],w=[2,4,8,16];for(let p=0;p<r.decoder_stages;p++){let B=n.codec_stages[p],k=t.createCommandEncoder({label:`codec_stage${p}`});for(let v=0;v<r.decoder_layers_per_stage;v++){let S=B.transformer_layers[v],F=f*e*4;G.size<F&&(C.push(G),G=this.createGPUBuffer(F,"codec_tmp"));let T=G,M=f*e,O=this.packUniform([{u:M}]);m(k,a.batchedCopy,[b,T,O],[_(M,256)],`codec_s${p}_l${v}_copy_res`);let z=this.packUniform([{u:e},{f:r.norm_eps},{u:f}]),E=this.createGPUBuffer(F,"codec_attn_normed");m(k,a.batchedRmsNorm,[b,S.attn_norm,E,z],[f],`codec_s${p}_l${v}_attn_norm`);let L=this.createGPUBuffer(f*e*4,"codec_q"),N=this.createGPUBuffer(f*e*4,"codec_k"),K=this.createGPUBuffer(f*e*4,"codec_v"),X=this.packUniform([{u:e},{u:e},{u:f}]);m(k,a.batchedMatvecF16,[S.wq,E,L,X],[e,f],`codec_s${p}_l${v}_qproj`),m(k,a.batchedMatvecF16,[S.wk,E,N,X],[e,f],`codec_s${p}_l${v}_kproj`),m(k,a.batchedMatvecF16,[S.wv,E,K,X],[e,f],`codec_s${p}_l${v}_vproj`);let y=this.packUniform([{u:r.n_heads},{u:r.head_dim},{u:f},{f:r.qk_norm_eps}]);m(k,a.qkNorm,[L,S.q_norm,y],[_(f*r.n_heads,128)],`codec_s${p}_l${v}_qnorm`),m(k,a.qkNorm,[N,S.k_norm,y],[_(f*r.n_heads,128)],`codec_s${p}_l${v}_knorm`);let $=w[p],U=this.createGPUBuffer(r.n_heads*f*$*4,"codec_scores"),A=this.packUniform([{u:r.n_heads},{u:r.head_dim},{u:f},{u:$}]);m(k,a.alibiAttnScore,[L,N,U,A],[_($,64),f,r.n_heads],`codec_s${p}_l${v}_attn_score`);let W=this.packUniform([{u:r.n_heads},{u:f},{u:$}]);m(k,a.codecSoftmax,[U,W],[_(r.n_heads*f,64)],`codec_s${p}_l${v}_softmax`);let H=this.createGPUBuffer(f*e*4,"codec_attn_out"),Y=this.packUniform([{u:r.n_heads},{u:r.head_dim},{u:f},{u:$}]);m(k,a.codecAttnValue,[U,K,H,Y],[_(r.n_heads*r.head_dim,64),f],`codec_s${p}_l${v}_attn_val`);let ae=this.createGPUBuffer(F,"codec_wo_out");m(k,a.batchedMatvecF16,[S.wo,H,ae,X],[e,f],`codec_s${p}_l${v}_wo`);let se=this.packUniform([{u:e},{u:M}]);m(k,a.batchedLayerScale,[ae,S.attn_scale,T,b,se],[_(M,256)],`codec_s${p}_l${v}_attn_res`),m(k,a.batchedCopy,[b,T,O],[_(M,256)],`codec_s${p}_l${v}_copy_ffn_res`);let I=this.createGPUBuffer(F,"codec_ffn_normed");m(k,a.batchedRmsNorm,[b,S.ffn_norm,I,z],[f],`codec_s${p}_l${v}_ffn_norm`);let R=f*r.hidden_dim,V=this.createGPUBuffer(R*4,"codec_gate"),Z=this.createGPUBuffer(R*4,"codec_up"),J=this.packUniform([{u:r.hidden_dim},{u:e},{u:f}]);m(k,a.batchedMatvecF16,[S.w1,I,V,J],[r.hidden_dim,f],`codec_s${p}_l${v}_gate`),m(k,a.batchedMatvecF16,[S.w3,I,Z,J],[r.hidden_dim,f],`codec_s${p}_l${v}_up`);let te=Math.min(_(R,256),65535),ee=_(_(R,256),te),me=this.packUniform([{u:R},{u:te}]);m(k,a.batchedSwiGLU,[V,Z,me],[te,ee],`codec_s${p}_l${v}_swiglu`);let Q=this.createGPUBuffer(F,"codec_down"),fe=this.packUniform([{u:e},{u:r.hidden_dim},{u:f}]);m(k,a.batchedMatvecF16,[S.w2,V,Q,fe],[e,f],`codec_s${p}_l${v}_down`),m(k,a.batchedLayerScale,[Q,S.ffn_scale,T,b,se],[_(M,256)],`codec_s${p}_l${v}_ffn_res`),C.push(E,L,N,K,U,H,ae,I,V,Z,Q)}if(B.conv_w&&B.conv_scale&&h[p]>1){let v=f*h[p],S=this.createGPUBuffer(v*e*4,"codec_upsampled"),F=this.packUniform([{u:e},{u:e},{u:l[p]},{u:v},{u:h[p]}]);m(k,a.causalConvTranspose1d,[b,B.conv_w,B.conv_scale,S,F],[_(v,64),e],`codec_s${p}_conv_up`),t.queue.submit([k.finish()]),await t.queue.onSubmittedWorkDone(),g[`after_stage${p}_transformer`]=await this.readF32Array(b,f*e),C.push(b),b=S,f=v,g[`after_stage${p}_conv_up`]=await this.readF32Array(b,f*e)}else t.queue.submit([k.finish()]),await t.queue.onSubmittedWorkDone(),g[`after_stage${p}_transformer`]=await this.readF32Array(b,f*e);for(let v of C)v.destroy();C=[]}{let p=f,B=this.createGPUBuffer(p*r.patch_size*4,"codec_output"),k=this.packUniform([{u:e},{u:r.patch_size},{u:7},{u:p},{u:1}]),v=t.createCommandEncoder({label:"codec_output"});m(v,a.causalConv1d,[b,n.codec_output_conv_w,n.codec_output_conv_g,B,k],[_(p,64),r.patch_size],"codec_output_conv"),t.queue.submit([v.finish()]),await t.queue.onSubmittedWorkDone(),g.after_output_conv=await this.readF32Array(B,p*r.patch_size),g.audio=g.after_output_conv,B.destroy()}return u.destroy(),q.destroy(),P.destroy(),c.destroy(),d.destroy(),b.destroy(),G.destroy(),g}uploadArray(s){let i=this.device.createBuffer({size:s.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,mappedAtCreation:!0});return s instanceof Uint32Array?new Uint32Array(i.getMappedRange()).set(s):new Float32Array(i.getMappedRange()).set(s),i.unmap(),i}createGPUBuffer(s,i){return this.device.createBuffer({size:Math.max(s,4),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,label:i})}async readBuffer(s,i){let t=this.device,a=t.createBuffer({size:i,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),n=t.createCommandEncoder();n.copyBufferToBuffer(s,0,a,0,i),t.queue.submit([n.finish()]),await a.mapAsync(GPUMapMode.READ);let r=a.getMappedRange().slice(0);return a.unmap(),a.destroy(),r}async readU32(s){let i=await this.readBuffer(s,4);return new Uint32Array(i)[0]}async readF32Array(s,i){let t=await this.readBuffer(s,i*4);return new Float32Array(t)}async readU32Array(s,i){let t=await this.readBuffer(s,i*4);return new Uint32Array(t)}get isReady(){return this.device!==null&&this.modelBuffers!==null&&this.pipelines!==null}async debugRead(s,i=16){let t=this.workBuffers,a=t[s];if(!a)throw new Error(`Unknown buffer: ${s}. Available: ${Object.keys(t).join(", ")}`);return this.readF32Array(a,i)}async debugBackboneStep(s){let i=this.device.createCommandEncoder();this.backboneStep(i,s),this.device.queue.submit([i.finish()]),await this.device.queue.onSubmittedWorkDone(),this.position++;let t=this.workBuffers,a=await this.readF32Array(t.hidden,16),n=await this.readF32Array(t.normed,16),r=await this.readF32Array(t.logits,16),o=await this.readU32(t.argmax_result),e=await this.readF32Array(t.logits,1024),g=-1/0;for(let u=0;u<e.length;u++)e[u]>g&&(g=e[u]);return{hidden:a,normed:n,logits_first16:r,logits_max:g,argmax:o}}async debugBackboneLayerByLayer(s){let i=this.pipelines,t=this.workBuffers,a=this.modelBuffers,n=this.config.backbone,r=this.position,o=n.dim,e;{let c=this.device.createCommandEncoder();e=c.beginComputePass({label:"debug_embed"});let d=this.packUniform([{u:s},{u:o}]);this.dispatch(e,i.embeddingLookup,[a.tok_embeddings,t.hidden,d],[_(o,256)]),e.end(),this.device.queue.submit([c.finish()]),await this.device.queue.onSubmittedWorkDone()}let g=await this.readF32Array(t.hidden,o),u=[];for(let c=0;c<n.n_layers;c++){let d=a.backbone_layers[c],f=this.kvCaches[c];{let h=this.device.createCommandEncoder();e=h.beginComputePass({label:`debug_l${c}_attn_prep`});let l=this.packUniform([{u:o}]);this.dispatch(e,i.copyBuffer,[t.hidden,t.residual,l],[_(o,256)]);let w=this.packUniform([{u:o},{f:n.norm_eps}]);this.dispatch(e,i.rmsNorm,[t.hidden,d.attn_norm,t.normed,w],[1]),e.end(),this.device.queue.submit([h.finish()]),await this.device.queue.onSubmittedWorkDone()}let b=await this.readF32Array(t.normed,o);{let h=this.device.createCommandEncoder();e=h.beginComputePass({label:`debug_l${c}_qkv`});let l=this.packUniform([{u:n.n_heads*n.head_dim},{u:o}]);this.dispatch(e,i.matvecF16,[d.wq,t.normed,t.q,l],[n.n_heads*n.head_dim]);let w=this.packUniform([{u:n.n_kv_heads*n.head_dim},{u:o}]);this.dispatch(e,i.matvecF16,[d.wk,t.normed,t.k,w],[n.n_kv_heads*n.head_dim]),this.dispatch(e,i.matvecF16,[d.wv,t.normed,t.v,w],[n.n_kv_heads*n.head_dim]),e.end(),e=h.beginComputePass({label:`debug_l${c}_rope_attn`});let p=this.packUniform([{u:n.head_dim},{u:r},{u:n.n_heads},{f:n.rope_theta}]);this.dispatch(e,i.rope,[t.q,p],[_(n.n_heads*n.head_dim/2,64)]);let B=this.packUniform([{u:n.head_dim},{u:r},{u:n.n_kv_heads},{f:n.rope_theta}]);this.dispatch(e,i.rope,[t.k,B],[_(n.n_kv_heads*n.head_dim/2,64)]);let k=this.packUniform([{u:r},{u:n.n_kv_heads*n.head_dim}]);this.dispatch(e,i.kvCacheWrite,[t.k,t.v,f.k,f.v,k],[_(n.n_kv_heads*n.head_dim,256)]);let v=r+1,S=n.n_heads/n.n_kv_heads,F=this.packUniform([{u:n.n_heads},{u:n.n_kv_heads},{u:n.head_dim},{u:v},{u:S}]);this.dispatch(e,i.attnScore,[t.q,f.k,t.scores,F],[_(n.n_heads*v,64)]),e.end(),e=h.beginComputePass({label:`debug_l${c}_attn_out`});let T=this.packUniform([{u:n.n_heads},{u:v}]);this.dispatch(e,i.softmax,[t.scores,T],[n.n_heads]);let M=this.packUniform([{u:n.n_heads},{u:n.n_kv_heads},{u:n.head_dim},{u:v},{u:S}]);this.dispatch(e,i.attnValue,[t.scores,f.v,t.attn_out,M],[_(n.n_heads*n.head_dim,128)]),e.end(),e=h.beginComputePass({label:`debug_l${c}_wo`});let O=this.packUniform([{u:o},{u:n.n_heads*n.head_dim}]);this.dispatch(e,i.matvecF16,[d.wo,t.attn_out,t.hidden,O],[o]),e.end(),e=h.beginComputePass({label:`debug_l${c}_res1`});let z=this.packUniform([{u:o}]);this.dispatch(e,i.addInPlace,[t.hidden,t.residual,z],[_(o,256)]),e.end(),this.device.queue.submit([h.finish()]),await this.device.queue.onSubmittedWorkDone()}let G=await this.readF32Array(t.hidden,o);{let h=this.device.createCommandEncoder();e=h.beginComputePass({label:`debug_l${c}_ffn_prep`});let l=this.packUniform([{u:o}]);this.dispatch(e,i.copyBuffer,[t.hidden,t.residual,l],[_(o,256)]);let w=this.packUniform([{u:o},{f:n.norm_eps}]);this.dispatch(e,i.rmsNorm,[t.hidden,d.ffn_norm,t.normed,w],[1]),e.end(),this.device.queue.submit([h.finish()]),await this.device.queue.onSubmittedWorkDone()}let C=await this.readF32Array(t.normed,o);{let h=this.device.createCommandEncoder();e=h.beginComputePass({label:`debug_l${c}_ffn`});let l=this.packUniform([{u:n.hidden_dim},{u:o}]);this.dispatch(e,i.matvecF16,[d.w1,t.normed,t.gate,l],[n.hidden_dim]),this.dispatch(e,i.matvecF16,[d.w3,t.normed,t.up,l],[n.hidden_dim]),e.end(),e=h.beginComputePass({label:`debug_l${c}_ffn_out`});let w=this.packUniform([{u:n.hidden_dim}]);this.dispatch(e,i.swiGLU,[t.gate,t.up,w],[_(n.hidden_dim,256)]);let p=this.packUniform([{u:o},{u:n.hidden_dim}]);this.dispatch(e,i.matvecF16,[d.w2,t.gate,t.hidden,p],[o]),e.end(),e=h.beginComputePass({label:`debug_l${c}_res2`});let B=this.packUniform([{u:o}]);this.dispatch(e,i.addInPlace,[t.hidden,t.residual,B],[_(o,256)]),e.end(),this.device.queue.submit([h.finish()]),await this.device.queue.onSubmittedWorkDone()}let m=await this.readF32Array(t.hidden,o);u.push({attn_norm:b,attn_out:G,ffn_norm:C,ffn_out:m})}{let c=this.device.createCommandEncoder();e=c.beginComputePass({label:"debug_final_norm"});let d=this.packUniform([{u:o},{f:n.norm_eps}]);this.dispatch(e,i.rmsNorm,[t.hidden,a.final_norm,t.normed,d],[1]),e.end(),this.device.queue.submit([c.finish()]),await this.device.queue.onSubmittedWorkDone()}let q=await this.readF32Array(t.normed,o),P=await this.readF32Array(t.hidden,o);return this.position++,{embed:g,layers:u,final_norm:q,hidden:P}}async debugFMForward(s=42){let i=this.config.fm,t=this.device.createCommandEncoder();this.fmForward(t),this.device.queue.submit([t.finish()]),await this.device.queue.onSubmittedWorkDone();let a=this.workBuffers,n=await this.readF32Array(a.semantic_logits,i.semantic_vocab),r=await this.readU32Array(a.acoustic_codes,i.n_acoustic_out),o=await this.readF32Array(a.x_t,i.n_acoustic_out);return{semantic_logits:n,velocities:[],acoustic_codes:r,x_final:o}}reset(){this.position=0}async backboneStepAndRead(s,i=!1){let t=this.device.createCommandEncoder();this.backboneStep(t,s,i),this.device.queue.submit([t.finish()]),await this.device.queue.onSubmittedWorkDone();let a=await this.readU32(this.workBuffers.argmax_result);return this.position++,a}async debugBackboneStepFull(s,i=!1){let t=this.device.createCommandEncoder();this.backboneStep(t,s,i),this.device.queue.submit([t.finish()]),await this.device.queue.onSubmittedWorkDone();let a=await this.readF32Array(this.workBuffers.normed,this.config.backbone.dim);return this.position++,a}async debugFMStep(s){let i=this.config.fm,t=this.device.createCommandEncoder();this.fmForward(t,s),this.device.queue.submit([t.finish()]),await this.device.queue.onSubmittedWorkDone();let a=this.workBuffers;return{semantic_logits:await this.readF32Array(a.semantic_logits,i.semantic_vocab),acoustic_codes:await this.readU32Array(a.acoustic_codes,i.n_acoustic_out),x_final:await this.readF32Array(a.x_t,i.n_acoustic_out)}}async fmStepAndRead(){let s=this.device.createCommandEncoder();return this.fmForward(s),this.device.queue.submit([s.finish()]),await this.device.queue.onSubmittedWorkDone(),this.readU32Array(this.workBuffers.acoustic_codes,this.config.fm.n_acoustic_out)}async generate(s,i,t,a,n=500,r,o){if(!this.isReady)throw new Error("Engine not initialized. Call init() and loadWeights() first.");this.reset();let e=performance.now();o?.("backbone");let g=[];if(a&&t>0){let l=this.config.backbone.dim;for(let w=0;w<t;w++){let p=this.device.createBuffer({size:l*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,mappedAtCreation:!0});new Float32Array(p.getMappedRange()).set(a.subarray(w*l,(w+1)*l)),p.unmap(),g.push(p)}}for(let l=0;l<s.length;l++){let w=s[l],p=this.device.createCommandEncoder();if(l>=i&&l<i+t&&g.length>0){let B=l-i;this.backboneStep(p,w,!1,g[B])}else this.backboneStep(p,w);this.device.queue.submit([p.finish()]),this.position++}await this.device.queue.onSubmittedWorkDone();let u=performance.now();o?.("backbone",u-e),o?.("fm");{let l=this.device.createCommandEncoder();this.backboneStep(l,24,!1),this.device.queue.submit([l.finish()]),await this.device.queue.onSubmittedWorkDone(),this.position++}let q=[],P=[],c=this.config.backbone,d=this.config.fm,f=this.pipelines,b=this.workBuffers,G=this.modelBuffers;for(let l=0;l<n;l++){if(l>0){let F=this.device.createCommandEncoder(),T=F.beginComputePass({label:`multiCBEmbed_frame${l}`}),M=this.packUniform([{u:c.dim},{u:8194},{u:23},{u:36}]);this.dispatch(T,f.multiCodebookEmbed,[G.audio_embeddings,b.semantic_argmax,b.acoustic_codes,b.hidden,M],[_(c.dim,256)]),T.end();let O=F.beginComputePass({label:`mcb_copy_frame${l}`}),z=this.packUniform([{u:c.dim}]);this.dispatch(O,f.copyBuffer,[b.hidden,b.fm_gate,z],[_(c.dim,256)]),O.end(),this.backboneStep(F,0,!1,b.fm_gate),this.device.queue.submit([F.finish()]),await this.device.queue.onSubmittedWorkDone(),this.position++}let w=this.device.createCommandEncoder();this.fmForward(w),this.device.queue.submit([w.finish()]),await this.device.queue.onSubmittedWorkDone();let p=await this.readF32Array(b.semantic_logits,d.semantic_vocab);p[0]=-1/0;let B=8194;for(let F=B;F<p.length;F++)p[F]=-1/0;let k=Ea(p,.9,.8);if(k<=1)break;q.push(k);let v=new Uint32Array([k]);this.device.queue.writeBuffer(b.semantic_argmax,0,v);let S=await this.readU32Array(b.acoustic_codes,d.n_acoustic_out);P.push(Array.from(S)),r?.(l,k,S)}let C=performance.now();o?.("fm",C-u),o?.("codec");let m;if(q.length>0){let l=new Uint32Array(q),w=new Uint32Array(P.flat());m=await this.codecDecode(l,w)}else m=new Float32Array(0);let h=performance.now();return o?.("codec",h-C),{semanticCodes:q,acousticCodes:P,audio:m,stats:{backboneMs:u-e,fmMs:C-u,codecMs:h-C,totalMs:h-e,framesGenerated:q.length}}}destroy(){if(this.workBuffers)for(let s of Object.values(this.workBuffers))s.destroy();for(let s of this.kvCaches)s.k.destroy(),s.v.destroy();this.device?.destroy()}};var j={UNK:0,BOS:1,EOS:2,INST:3,INST_END:4,AUDIO:24,BEGIN_AUDIO:25,OUTPUT_AUDIO:26,AUDIO_TO_TEXT:35,TEXT_TO_AUDIO:36,PAD:11},ie=class x{bytesToRank=new Map;specialTokens=new Map;pattern;voiceNumTokens=new Map;numSpecialTokens;constructor(s){this.numSpecialTokens=s.config.default_num_special_tokens;for(let i of s.vocab){let t=atob(i.token_bytes);this.bytesToRank.set(t,i.rank)}for(let i of s.special_tokens)this.specialTokens.set(i.token_str,i.rank);try{this.pattern=new RegExp(s.config.pattern,"gu")}catch{this.pattern=/\S+|\s+/gu}if(s.audio?.voice_num_audio_tokens)for(let[i,t]of Object.entries(s.audio.voice_num_audio_tokens))this.voiceNumTokens.set(i,t)}static async load(s){let t=await(await fetch(s)).json();return new x(t)}getVoiceNumTokens(s){let i=this.voiceNumTokens.get(s);if(i===void 0)throw new Error(`Unknown voice: ${s}. Available: ${[...this.voiceNumTokens.keys()].join(", ")}`);return i}buildTTSPrompt(s,i){let t=this.getVoiceNumTokens(i),a=[];a.push(j.BOS),a.push(j.BEGIN_AUDIO);let n=a.length;for(let o=0;o<t;o++)a.push(j.AUDIO);a.push(j.TEXT_TO_AUDIO);let r=this.encode(s);return a.push(...r),a.push(j.AUDIO_TO_TEXT),a.push(j.BEGIN_AUDIO),{tokens:a,audioTokenStart:n,audioTokenCount:t}}encode(s){let i=[],t=s.matchAll(this.pattern);for(let a of t){let n=a[0],r=this.bpeEncode(n);i.push(...r)}return i}bpeEncode(s){let t=new TextEncoder().encode(s),a=[];for(let n of t)a.push(String.fromCharCode(n));if(a.length<=1){let n=this.bytesToRank.get(a[0]);return[n!==void 0?n+this.numSpecialTokens:j.UNK]}for(;a.length>1;){let n=1/0,r=-1;for(let e=0;e<a.length-1;e++){let g=a[e]+a[e+1],u=this.bytesToRank.get(g);u!==void 0&&u<n&&(n=u,r=e)}if(r===-1)break;let o=[];for(let e=0;e<a.length;e++)e===r?(o.push(a[e]+a[e+1]),e++):o.push(a[e]);a=o}return a.map(n=>{let r=this.bytesToRank.get(n);return r!==void 0?r+this.numSpecialTokens:j.UNK})}get voices(){return[...this.voiceNumTokens.keys()]}};var La="https://huggingface.co/mistralai/Voxtral-4B-TTS-2603/resolve/main",ve=class x{engine;tokenizer;modelsUrl;voiceCache=new Map;constructor(s,i,t){this.engine=s,this.tokenizer=i,this.modelsUrl=t}static async load(s={}){let i=s.modelsUrl??La,t=new oe({maxSeqLen:s.maxSeqLen});await t.init();let[,a]=await Promise.all([t.loadWeightsFromHF(s.weightsUrl??re,s.onProgress),ie.load(`${i}/tekken.json`)]);return new x(t,a,i)}get voices(){return this.tokenizer.voices}async speak(s,i="casual_female",t={}){let{tokens:a,audioTokenStart:n,audioTokenCount:r}=this.tokenizer.buildTTSPrompt(s,i),o=this.voiceCache.get(i);if(!o){let e=await fetch(`${this.modelsUrl}/voice_embedding_f32/${i}.bin`);if(!e.ok)throw new Error(`Failed to load voice "${i}": ${e.status} ${e.statusText}`);o=new Float32Array(await e.arrayBuffer()),this.voiceCache.set(i,o)}return this.engine.generate(a,n,r,o,t.maxFrames??500,t.onFrame,t.onStage)}destroy(){this.engine.destroy(),this.voiceCache.clear()}};export{re as HF_VOXTRAL_URL,j as TOKENS,ie as TekkenTokenizer,ve as Voxtral,oe as VoxtralEngine,za as clearWeightCache,Ma as getWeightCacheInfo};
