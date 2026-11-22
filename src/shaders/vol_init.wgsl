// rest volume + mass from how crowded the splats are
//#include "sim.wgsl"
@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> cells: array<vec4i>;
@group(0) @binding(2) var<uniform> u: SimU;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= u.n) {
    return;
  }
  let p = particles[id.x];
  let ci = vec3i(floor(p.pos));
  var w = weights(p.pos - (vec3f(ci) + 0.5));
  var density = 0.0;
  for (var gx = 0; gx < 3; gx++) {
    for (var gy = 0; gy < 3; gy++) {
      for (var gz = 0; gz < 3; gz++) {
        let wt = w[gx].x * w[gy].y * w[gz].z;
        density +=
          f32(cells[cellIndex(ci + vec3i(gx - 1, gy - 1, gz - 1), u.grid)].w) / COUNT_FP * wt;
      }
    }
  }
  // every splat starts with mass 1, so this is splats per cell. the floor stops lone
  // splats getting a huge volume and blowing up
  let vol = 1.0 / max(density, 3.0);
  particles[id.x].vol = vol;
  // same density everywhere, or a crowded surface outweighs what holds it up
  particles[id.x].mass = vol * 8.0;
}
