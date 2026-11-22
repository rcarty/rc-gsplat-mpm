// positions are in grid units

// the splat-count pass scatters with this, not FP. at FP a cell overflows i32 past ~2,000 splats,
// which a dense scan can reach
const COUNT_FP: f32 = 1024.0;

//#include "../../kit/src/mpm/common.wgsl"

struct Particle {
  pos: vec3f,
  mass: f32,
  v: vec3f,
  vol: f32,
  C: mat3x3f,
  F: mat3x3f,
  tau: mat3x3f,
  Jp: f32,
}

struct SimU {
  grid: vec3i,
  n: u32,
  dt: f32,
  gravity: f32,
  mu0: f32,
  la0: f32,
  material: u32,
  hardening: f32,
  alphaDP: f32,
  floorY: f32,
  friction: f32,
  pokeR: f32,
  viscosity: f32,
  pokeO: vec4f,
  pokeD: vec4f,
  pokeV: vec4f,
}
