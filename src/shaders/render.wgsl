struct RU {
  view: mat4x4f,
  proj: mat4x4f,
  size: vec2f,
  focal: vec2f,
  map: vec4f,
  depth: vec4f,
}

// a = centre px + radius, b = conic + opacity, c = colour
struct SplatOut {
  a: vec4f,
  b: vec4f,
  c: vec4f,
}
