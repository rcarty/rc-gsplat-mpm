// grid floor at y = 0
//#include "../../kit/src/render/fullscreen.wgsl"
struct FU {
  invVP: mat4x4f,
  size: vec4f,
}

@group(0) @binding(0) var<uniform> f: FU;

@fragment
fn fs(@builtin(position) fc: vec4f) -> @location(0) vec4f {
  let bg = vec3f(0.051, 0.055, 0.067);
  let ndc = vec2f(fc.x / f.size.x * 2.0 - 1.0, 1.0 - fc.y / f.size.y * 2.0);
  let a = f.invVP * vec4f(ndc, 0.0, 1.0);
  let b = f.invVP * vec4f(ndc, 1.0, 1.0);
  let o = a.xyz / a.w;
  let dir = normalize(b.xyz / b.w - o);
  // fwidth needs uniform control flow, so do the hit for every pixel and mask after
  let below = dir.y < -1e-4;
  let hit = o + dir * (-o.y / min(dir.y, -1e-4));
  let g = hit.xz * 4.0;
  let fw = max(fwidth(g), vec2f(1e-4));
  let lines = abs(fract(g - 0.5) - 0.5) / fw;
  let line = 1.0 - min(min(lines.x, lines.y), 1.0);
  let fade = select(0.0, exp(-length(hit.xz) * 0.45), below);
  return vec4f(bg + vec3f(0.035, 0.04, 0.05) * fade + vec3f(0.07, 0.085, 0.1) * line * fade, 1.0);
}
