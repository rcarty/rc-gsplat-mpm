// mass only, run once after a reset before vol_init
//#include "sim.wgsl"
//#include "../../kit/src/mpm/cell.wgsl"
@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> cells: array<Cell>;
@group(0) @binding(2) var<uniform> u: SimU;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= u.n) {
    return;
  }
  let p = particles[id.x];
  let ci = vec3i(floor(p.pos));
  var w = weights(p.pos - (vec3f(ci) + 0.5));
  for (var gx = 0; gx < 3; gx++) {
    for (var gy = 0; gy < 3; gy++) {
      for (var gz = 0; gz < 3; gz++) {
        let wt = w[gx].x * w[gy].y * w[gz].z;
        atomicAdd(
          &cells[cellIndex(ci + vec3i(gx - 1, gy - 1, gz - 1), u.grid)].m,
          i32(wt * p.mass * COUNT_FP),
        );
      }
    }
  }
}
