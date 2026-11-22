// scatter mass + momentum, stress from the last g2p goes in the affine term
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
  let affine = p.tau * (-u.dt * p.vol * 4.0) + p.C * p.mass;
  for (var gx = 0; gx < 3; gx++) {
    for (var gy = 0; gy < 3; gy++) {
      for (var gz = 0; gz < 3; gz++) {
        let wt = w[gx].x * w[gy].y * w[gz].z;
        let cx = ci + vec3i(gx - 1, gy - 1, gz - 1);
        let dist = (vec3f(cx) - p.pos) + 0.5;
        let mv = wt * (p.mass * p.v + affine * dist);
        let idx = cellIndex(cx, u.grid);
        atomicAdd(&cells[idx].m, i32(wt * p.mass * FP));
        atomicAdd(&cells[idx].vx, i32(mv.x * FP));
        atomicAdd(&cells[idx].vy, i32(mv.y * FP));
        atomicAdd(&cells[idx].vz, i32(mv.z * FP));
      }
    }
  }
}
