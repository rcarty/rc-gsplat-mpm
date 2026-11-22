//#include "sim.wgsl"
@group(0) @binding(0) var<storage, read> cells: array<vec4i>;
@group(0) @binding(1) var<storage, read_write> gridV: array<vec4f>;
@group(0) @binding(2) var<uniform> u: SimU;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let g = u.grid;
  if (id.x >= u32(g.x * g.y * g.z)) {
    return;
  }
  let c = cells[id.x];
  let mass = f32(c.w) / FP;
  var v = vec3f(0.0);
  if (mass > 1e-4) {
    v = vec3f(c.xyz) / FP / mass;
    v.y -= u.gravity * u.dt;
    let i = i32(id.x);
    let x = i % g.x;
    let y = (i / g.x) % g.y;
    let z = i / (g.x * g.y);
    if (f32(y) < u.floorY && v.y < 0.0) {
      // floor, Coulomb friction
      let vn = -v.y;
      v.y = 0.0;
      let lt = length(v.xz);
      if (lt > 1e-6) {
        v =
          vec3f(
            v.x * max(0.0, 1.0 - u.friction * vn / lt),
            0.0,
            v.z * max(0.0, 1.0 - u.friction * vn / lt),
          );
      }
    }
    if (y > g.y - 4 && v.y > 0.0) {
      v.y = 0.0;
    }
    if ((x < 3 && v.x < 0.0) || (x > g.x - 4 && v.x > 0.0)) {
      v.x = 0.0;
    }
    if ((z < 3 && v.z < 0.0) || (z > g.z - 4 && v.z > 0.0)) {
      v.z = 0.0;
    }
  }
  gridV[id.x] = vec4f(v, mass);
}
