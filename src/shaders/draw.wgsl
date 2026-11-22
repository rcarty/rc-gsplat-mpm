//#include "render.wgsl"
@group(0) @binding(0) var<storage, read> outs: array<SplatOut>;
@group(0) @binding(1) var<storage, read> order: array<u32>;
@group(0) @binding(2) var<uniform> r: RU;

struct V {
  @builtin(position) pos: vec4f,
  @location(0) d: vec2f,
  @location(1) conic: vec4f,
  @location(2) col: vec3f,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> V {
  var corners =
    array<vec2f, 6>(
      vec2f(-1.0, -1.0),
      vec2f(1.0, -1.0),
      vec2f(-1.0, 1.0),
      vec2f(-1.0, 1.0),
      vec2f(1.0, -1.0),
      vec2f(1.0, 1.0),
    );
  let o = outs[order[ii]];
  var out: V;
  if (o.a.z <= 0.0) {
    out.pos = vec4f(2.0, 2.0, 2.0, 1.0);
    return out;
  }
  let d = corners[vi] * o.a.z;
  let px = o.a.xy + d;
  out.pos = vec4f(px / r.size * 2.0 - 1.0, 0.0, 1.0);
  out.d = d;
  out.conic = o.b;
  out.col = o.c.rgb;
  return out;
}

@fragment
fn fs(i: V) -> @location(0) vec4f {
  let d = i.d;
  let power = -0.5 * (i.conic.x * d.x * d.x + i.conic.z * d.y * d.y) - i.conic.y * d.x * d.y;
  if (power > 0.0) {
    discard;
  }
  let a = min(0.99, i.conic.w * exp(power));
  if (a < 1.0 / 255.0) {
    discard;
  }
  return vec4f(i.col * a, a);
}
