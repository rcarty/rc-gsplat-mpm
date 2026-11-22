// gather, advect, update F, apply the material. tau gets scattered by the next p2g
//#include "sim.wgsl"
//#include "svd.wgsl"
@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> gridV: array<vec4f>;
@group(0) @binding(2) var<uniform> u: SimU;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= u.n) {
    return;
  }
  let p = particles[id.x];
  let ci = vec3i(floor(p.pos));
  var w = weights(p.pos - (vec3f(ci) + 0.5));
  var nv = vec3f(0.0);
  var B = mat3x3f(vec3f(0.0), vec3f(0.0), vec3f(0.0));
  for (var gx = 0; gx < 3; gx++) {
    for (var gy = 0; gy < 3; gy++) {
      for (var gz = 0; gz < 3; gz++) {
        let wt = w[gx].x * w[gy].y * w[gz].z;
        let cx = ci + vec3i(gx - 1, gy - 1, gz - 1);
        let dist = (vec3f(cx) - p.pos) + 0.5;
        let gv = gridV[cellIndex(cx, u.grid)].xyz;
        nv += gv * wt;
        B += mat3x3f(gv * dist.x, gv * dist.y, gv * dist.z) * wt;
      }
    }
  }
  let C = B * 4.0;
  // poke
  nv += u.pokeV.xyz * rayFalloff(p.pos, u.pokeO.xyz, u.pokeD.xyz, u.pokeR);
  let sp = length(nv);
  if (sp > 3.0) {
    nv *= 3.0 / sp;
  }
  let pos = clamp(p.pos + nv * u.dt, vec3f(1.0), vec3f(u.grid) - 2.0);

  var F = (I3 + C * u.dt) * p.F;
  var Jp = p.Jp;
  var tau = mat3x3f(vec3f(0.0), vec3f(0.0), vec3f(0.0));
  var mu = u.mu0;
  var la = u.la0;
  if (u.material == 3u) {
    // liquid: keep the volume change only
    let J = clamp(determinant(F), 0.6, 1.4);
    F = I3 * pow(J, 1.0 / 3.0);
    tau = I3 * (la * J * (J - 1.0)) + (C + transpose(C)) * u.viscosity;
  } else {
    let d = svd3(F);
    var s = d.s;
    if (u.material == 1u) {
      // sand: Drucker-Prager in Hencky strain
      var e = log(max(s, vec3f(1e-3)));
      let tr = e.x + e.y + e.z;
      if (tr >= 0.0) {
        e = vec3f(0.0);
      } else {
        let eh = e - vec3f(tr / 3.0);
        let ehn = length(eh);
        let dg = ehn + (3.0 * la + 2.0 * mu) / (2.0 * mu) * tr * u.alphaDP;
        if (dg > 0.0 && ehn > 1e-8) {
          e = e - dg * eh / ehn;
        }
      }
      s = exp(e);
      F = d.U * diag(s) * transpose(d.V);
      let tre = e.x + e.y + e.z;
      tau = d.U * diag(2.0 * mu * e + vec3f(la * tre)) * transpose(d.U);
    } else {
      if (u.material == 2u) {
        // snow: clamp stretch, harden under compression
        let ns = clamp(s, vec3f(1.0 - 0.025), vec3f(1.0 + 0.0075));
        Jp = clamp(Jp * (s.x * s.y * s.z) / (ns.x * ns.y * ns.z), 0.5, 10.0);
        s = ns;
        F = d.U * diag(s) * transpose(d.V);
        let h = clamp(exp(u.hardening * (1.0 - Jp)), 0.1, 5.0);
        mu = mu * h;
        la = la * h;
      }
      if (u.material == 0u) {
        // jelly: cap the stretch so one bad step can't run away
        s = clamp(s, vec3f(0.3), vec3f(3.0));
        F = d.U * diag(s) * transpose(d.V);
      }
      // fixed-corotated
      let R = d.U * transpose(d.V);
      let J = s.x * s.y * s.z;
      tau = (F - R) * transpose(F) * (2.0 * mu) + I3 * (la * J * (J - 1.0));
    }
  }
  particles[id.x].pos = pos;
  particles[id.x].v = nv;
  particles[id.x].C = C;
  particles[id.x].F = F;
  particles[id.x].tau = tau;
  particles[id.x].Jp = Jp;
}
