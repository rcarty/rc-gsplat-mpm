// deform the covariance (F S0 F^T), EWA project, write a 16-bit depth key
//#include "sim.wgsl"
//#include "render.wgsl"
struct Rest {
  covA: vec4f,
  covB: vec4f,
  color: vec4f,
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> rest: array<Rest>;
@group(0) @binding(2) var<storage, read_write> outs: array<SplatOut>;
@group(0) @binding(3) var<storage, read_write> keys: array<u32>;
@group(0) @binding(4) var<storage, read_write> vals: array<u32>;
@group(0) @binding(5) var<uniform> r: RU;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= u32(r.depth.z)) {
    return;
  }
  vals[i] = i;
  let P = particles[i];
  let R = rest[i];
  let S0 =
    mat3x3f(
      R.covA.x,
      R.covA.y,
      R.covA.z,
      R.covA.y,
      R.covA.w,
      R.covB.x,
      R.covA.z,
      R.covB.x,
      R.covB.y,
    );
  let s = r.map.x;
  let world = vec3f(P.pos.x - r.map.y, P.pos.y - r.map.z, P.pos.z - r.map.w) * s;
  let tv = (r.view * vec4f(world, 1.0)).xyz;
  let d = -tv.z;
  outs[i].a = vec4f(0.0);
  keys[i] = 0xFFFFu;
  if (d < 0.05) {
    return;
  }
  let W = mat3x3f(r.view[0].xyz, r.view[1].xyz, r.view[2].xyz);
  let M = W * (P.F * S0 * transpose(P.F) * (s * s)) * transpose(W);
  let j0 = vec3f(r.focal.x / d, 0.0, r.focal.x * tv.x / (d * d));
  let j1 = vec3f(0.0, r.focal.y / d, r.focal.y * tv.y / (d * d));
  let c00 = dot(j0, M * j0) + 0.3;
  let c01 = dot(j0, M * j1);
  let c11 = dot(j1, M * j1) + 0.3;
  let det = c00 * c11 - c01 * c01;
  if (det <= 1e-8) {
    return;
  }
  let mid = 0.5 * (c00 + c11);
  let rad = 3.0 * sqrt(mid + sqrt(max(0.1, mid * mid - det)));
  if (!(rad < 512.0)) {
    return; // too big to be right, and a clipped quad shows as a square
  }
  let clip = r.proj * vec4f(tv, 1.0);
  let px = (clip.xy / clip.w * 0.5 + 0.5) * r.size;
  if (px.x + rad < 0.0 || px.y + rad < 0.0 || px.x - rad > r.size.x || px.y - rad > r.size.y) {
    return;
  }
  outs[i] =
    SplatOut(
      vec4f(px, rad, 0.0),
      vec4f(c11 / det, -c01 / det, c00 / det, R.color.a),
      vec4f(R.color.rgb, 0.0),
    );
  let q = clamp((d - r.depth.x) / (r.depth.y - r.depth.x), 0.0, 1.0);
  keys[i] = u32((1.0 - q) * 65534.0); // far splats get small keys so they draw first
}
