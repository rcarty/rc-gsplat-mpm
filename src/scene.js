import { vec3 } from '../kit/src/index.js';

export const G = 64; // grid cells per side
export const FLOOR = 3; // floor height in cells

const { cross, dot, normalize } = vec3;

// covariance from 3 axes + scales, as [xx, xy, xz, yy, yz, zz]
function covFromAxes(a0, a1, a2, s) {
  const c = [0, 0, 0, 0, 0, 0];
  const ax = [a0, a1, a2];
  for (let k = 0; k < 3; k++) {
    const a = ax[k];
    const v = s[k] * s[k];
    c[0] += a[0] * a[0] * v;
    c[1] += a[0] * a[1] * v;
    c[2] += a[0] * a[2] * v;
    c[3] += a[1] * a[1] * v;
    c[4] += a[1] * a[2] * v;
    c[5] += a[2] * a[2] * v;
  }
  return c;
}

// value noise for the marble
function hash3(x, y, z) {
  let h = (x * 374761393 + y * 668265263 + z * 1274126177) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}
function noise3(x, y, z) {
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const zi = Math.floor(z);
  const xf = x - xi;
  const yf = y - yi;
  const zf = z - zi;
  const s = t => t * t * (3 - 2 * t);
  const u = s(xf);
  const v = s(yf);
  const w = s(zf);
  const L = (a, b, t) => a + (b - a) * t;
  const c = (i, j, k) => hash3(xi + i, yi + j, zi + k);
  return L(
    L(L(c(0, 0, 0), c(1, 0, 0), u), L(c(0, 1, 0), c(1, 1, 0), u), v),
    L(L(c(0, 0, 1), c(1, 0, 1), u), L(c(0, 1, 1), c(1, 1, 1), u), v),
    w,
  );
}
const fbm = (x, y, z) =>
  noise3(x, y, z) * 0.5 +
  noise3(x * 2.1, y * 2.1, z * 2.1) * 0.3 +
  noise3(x * 4.3, y * 4.3, z * 4.3) * 0.2;

// (2,3) torus knot covered in flat marble splats
export function makeKnot(count) {
  const P = 2;
  const Q = 3;
  const a = 0.42;
  const curve = th => {
    const r = 2 + Math.cos(Q * th);
    return [r * Math.cos(P * th), r * Math.sin(P * th), -Math.sin(Q * th) * 0.9];
  };
  const tangent = th => {
    const e = 1e-4;
    const p = curve(th + e);
    const m = curve(th - e);
    return [(p[0] - m[0]) / (2 * e), (p[1] - m[1]) / (2 * e), (p[2] - m[2]) / (2 * e)];
  };
  let maxSpeed = 0;
  for (let i = 0; i < 400; i++)
    maxSpeed = Math.max(maxSpeed, Math.hypot(...tangent((i / 400) * Math.PI * 2)));
  let length = 0;
  for (let i = 0; i < 2000; i++)
    length += Math.hypot(...tangent((i / 2000) * Math.PI * 2)) * ((Math.PI * 2) / 2000);
  const area = length * 2 * Math.PI * a;
  const spacing = Math.sqrt(area / count);
  const pos = new Float32Array(count * 3);
  const cov = new Float32Array(count * 6);
  const col = new Float32Array(count * 4);
  const L = normalize([0.45, 0.8, 0.55]);
  let i = 0;
  let seed = 7;
  const rnd = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  while (i < count) {
    const th = rnd() * Math.PI * 2;
    const T = tangent(th);
    if (rnd() > Math.hypot(...T) / maxSpeed) continue; // keeps density even along the tube
    const t = normalize(T);
    const N = normalize(cross(t, [0, 0, 1]));
    const Bv = cross(t, N);
    const ph = rnd() * Math.PI * 2;
    const c = curve(th);
    const n = [
      N[0] * Math.cos(ph) + Bv[0] * Math.sin(ph),
      N[1] * Math.cos(ph) + Bv[1] * Math.sin(ph),
      N[2] * Math.cos(ph) + Bv[2] * Math.sin(ph),
    ];
    const p = [c[0] + a * n[0], c[1] + a * n[1], c[2] + a * n[2]];
    const t2 = cross(n, t);
    pos.set(p, i * 3);
    cov.set(covFromAxes(t, t2, n, [spacing * 0.75, spacing * 0.75, spacing * 0.08]), i * 6);
    // pale stone, blue veins, lighting baked in like a real scan
    const vein = Math.pow(
      1 -
        Math.abs(
          Math.sin(
            p[0] * 0.9 + p[1] * 0.6 + p[2] * 0.4 + fbm(p[0] * 0.8 + 5, p[1] * 0.8, p[2] * 0.8) * 6,
          ),
        ),
      14,
    );
    const base = 0.9 + (fbm(p[0] * 6, p[1] * 6, p[2] * 6) - 0.5) * 0.12;
    let r = base * 0.93;
    let g = base * 0.92;
    let b = base * 0.9;
    r += (0.25 - r) * vein * 0.85;
    g += (0.5 - g) * vein * 0.85;
    b += (0.78 - b) * vein * 0.85;
    const lit = 0.38 + 0.55 * Math.max(0, dot(n, L)) + 0.12 * Math.max(0, n[1]);
    col.set([Math.min(1, r * lit), Math.min(1, g * lit), Math.min(1, b * lit), 0.95], i * 4);
    i++;
  }
  return { n: count, pos, cov, col };
}

// standard 3DGS .ply, binary little-endian only
export function parsePly(buffer, maxCount) {
  const head = new TextDecoder('latin1').decode(
    new Uint8Array(buffer, 0, Math.min(buffer.byteLength, 100000)),
  );
  const end = head.match(/end_header\r?\n/);
  if (!end) throw new Error('No PLY header found');
  const headEnd = end.index + end[0].length;
  const header = head.slice(0, headEnd);
  if (!/format binary_little_endian/.test(header))
    throw new Error('Only binary little-endian PLY files are supported');
  const count = +((header.match(/element vertex (\d+)/) || [])[1] || 0);
  if (!count) throw new Error('No vertices in this file');
  const sizes = {
    char: 1,
    int8: 1,
    uchar: 1,
    uint8: 1,
    short: 2,
    int16: 2,
    ushort: 2,
    uint16: 2,
    int: 4,
    int32: 4,
    uint: 4,
    uint32: 4,
    float: 4,
    float32: 4,
    double: 8,
    float64: 8,
  };
  const props = [];
  let stride = 0;
  let inVertex = false;
  for (const line of header.split('\n')) {
    const w = line.trim().split(/\s+/);
    if (w[0] === 'element') inVertex = w[1] === 'vertex';
    else if (w[0] === 'property' && inVertex) {
      // a wrong size here shifts every splat after it, so don't guess
      if (!sizes[w[1]]) throw new Error(`Unsupported PLY property type ${w[1]}`);
      props.push({ name: w[2], type: w[1], off: stride });
      stride += sizes[w[1]];
    }
  }
  if (headEnd + count * stride > buffer.byteLength) throw new Error('The file is cut short');
  const off = name => {
    const p = props.find(p => p.name === name);
    return p && (p.type === 'float' || p.type === 'float32') ? p.off : -1;
  };
  const need = [
    'x',
    'y',
    'z',
    'f_dc_0',
    'f_dc_1',
    'f_dc_2',
    'opacity',
    'scale_0',
    'scale_1',
    'scale_2',
    'rot_0',
    'rot_1',
    'rot_2',
    'rot_3',
  ];
  const o = Object.fromEntries(need.map(n => [n, off(n)]));
  for (const n of need)
    if (o[n] < 0) throw new Error(`This doesn't look like a 3DGS file (missing ${n})`);
  const dv = new DataView(buffer, headEnd);
  const alphaOf = b => 1 / (1 + Math.exp(-dv.getFloat32(b + o.opacity, true)));

  // too many splats: keep big opaque ones more often, then grow the survivors to cover the gaps
  const n = Math.min(count, maxCount);
  let pick = null;
  let grow = 1;
  if (count > n) {
    const key = new Float32Array(count);
    const weight = new Float32Array(count);
    let all = 0;
    let x = 2463534242;
    for (let i = 0; i < count; i++) {
      const b = i * stride;
      const e = [0, 1, 2].map(k => Math.exp(dv.getFloat32(b + o['scale_' + k], true)));
      const w = (alphaOf(b) * e[0] * e[1] * e[2]) / Math.min(e[0], e[1], e[2]); // opacity * area
      weight[i] = Number.isFinite(w) ? w : 0;
      all += weight[i];
      x ^= x << 13;
      x ^= x >>> 17;
      x ^= x << 5;
      key[i] = Math.log(((x >>> 0) + 1) / 4294967297) / weight[i]; // weighted sampling, largest keys win
    }
    const cut = key.slice().sort()[count - n];
    pick = new Uint32Array(n);
    let kept = 0;
    let m = 0;
    for (let i = 0; i < count && m < n; i++) {
      if (key[i] >= cut) {
        pick[m++] = i;
        kept += weight[i];
      }
    }
    grow = Math.min(all / kept, count / n);
  }
  const pos = new Float32Array(n * 3);
  const cov = new Float32Array(n * 6);
  const col = new Float32Array(n * 4);
  const SH0 = 0.28209479177387814;
  for (let i = 0; i < n; i++) {
    const b = (pick ? pick[i] : i) * stride;
    const f = k => dv.getFloat32(b + o[k], true);
    // 3DGS is y-down, flip to y-up
    pos.set([f('x'), -f('y'), -f('z')], i * 3);
    let qw = f('rot_0');
    let qx = f('rot_1');
    let qy = f('rot_2');
    let qz = f('rot_3');
    const ql = Math.hypot(qw, qx, qy, qz) || 1;
    qw /= ql;
    qx /= ql;
    qy /= ql;
    qz /= ql;
    const R = [
      [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy + qw * qz), 2 * (qx * qz - qw * qy)],
      [2 * (qx * qy - qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz + qw * qx)],
      [2 * (qx * qz + qw * qy), 2 * (qy * qz - qw * qx), 1 - 2 * (qx * qx + qy * qy)],
    ];
    const flip = a => [a[0], -a[1], -a[2]];
    cov.set(
      covFromAxes(flip(R[0]), flip(R[1]), flip(R[2]), [
        Math.exp(f('scale_0')),
        Math.exp(f('scale_1')),
        Math.exp(f('scale_2')),
      ]).map(v => v * grow),
      i * 6,
    );
    const c = k => Math.min(1, Math.max(0, 0.5 + SH0 * f(k)));
    col.set([c('f_dc_0'), c('f_dc_1'), c('f_dc_2'), alphaOf(b)], i * 4);
  }
  return { n, pos, cov, col, total: count };
}

// fit splats into the grid, then fill the inside
export function buildParticles(s, dropHeight) {
  // 2nd-98th percentile bounds to drop floaters
  const pct = (axis, q) => {
    const v = new Float32Array(s.n);
    for (let i = 0; i < s.n; i++) v[i] = s.pos[i * 3 + axis];
    v.sort();
    return v[Math.floor(q * (s.n - 1))];
  };
  const lo = [0, 1, 2].map(k => pct(k, 0.02));
  const hi = [0, 1, 2].map(k => pct(k, 0.98));
  const ext = [0, 1, 2].map(k => hi[k] - lo[k]);
  const pad = ext.map(e => e * 0.1);
  const keep = [];
  for (let i = 0; i < s.n; i++) {
    let ok = s.col[i * 4 + 3] > 0.02;
    for (let k = 0; k < 3 && ok; k++) {
      const v = s.pos[i * 3 + k];
      ok = v >= lo[k] - pad[k] && v <= hi[k] + pad[k];
    }
    if (ok) keep.push(i);
  }
  const target = 32; // cells across the longest side
  const scale = target / Math.max(ext[0] + 2 * pad[0], ext[1] + 2 * pad[1], ext[2] + 2 * pad[2]);
  const ctr = [(lo[0] + hi[0]) / 2, lo[1] - pad[1], (lo[2] + hi[2]) / 2];
  const toGrid = (x, y, z) => [
    (x - ctr[0]) * scale + G / 2,
    (y - ctr[1]) * scale + FLOOR + dropHeight,
    (z - ctr[2]) * scale + G / 2,
  ];

  const nSurf = keep.length;
  if (!nSurf || !Number.isFinite(scale)) throw new Error('Not enough splats to build an object');
  const surf = new Float32Array(nSurf * 3);
  const R2 = 2;
  const GR = G * R2;
  const NV = GR * GR * GR;
  const label = new Int32Array(NV).fill(-1); // nearest surface splat per voxel
  const vox = (x, y, z) =>
    Math.floor(x * R2) + Math.floor(y * R2) * GR + Math.floor(z * R2) * GR * GR;
  const queue = new Int32Array(NV);
  const surfOcc = new Uint8Array(NV);
  let qh = 0;
  let qt = 0;
  for (let k = 0; k < nSurf; k++) {
    const i = keep[k];
    const p = toGrid(s.pos[i * 3], s.pos[i * 3 + 1], s.pos[i * 3 + 2]);
    for (let a = 0; a < 3; a++) p[a] = Math.min(G - 2.01, Math.max(1.01, p[a]));
    surf.set(p, k * 3);
    const v = vox(p[0], p[1], p[2]);
    if (label[v] < 0) queue[qt++] = v;
    label[v] = k;
    surfOcc[v] = 1;
  }
  // scans have holes and no underside, so a flood fill leaks. a voxel counts as inside
  // when it can see the surface in at least 5 of the 6 axis directions
  const seen = new Uint8Array(NV);
  const sweep = (start, stride, len) => {
    let hit = false;
    for (let k = 0, i = start; k < len; k++, i += stride) {
      if (hit) seen[i]++;
      if (surfOcc[i]) hit = true;
    }
    hit = false;
    for (let k = 0, i = start + (len - 1) * stride; k < len; k++, i -= stride) {
      if (hit) seen[i]++;
      if (surfOcc[i]) hit = true;
    }
  };
  for (let a = 0; a < GR; a++)
    for (let b = 0; b < GR; b++) {
      sweep(a * GR + b * GR * GR, 1, GR);
      sweep(a + b * GR * GR, GR, GR);
      sweep(a + b * GR, GR * GR, GR);
    }
  const outside = new Uint8Array(NV);
  for (let i = 0; i < NV; i++) outside[i] = seen[i] < 5 && !surfOcc[i] ? 1 : 0;
  const nb = (i, f) => {
    const x = i % GR;
    const y = ((i / GR) | 0) % GR;
    const z = (i / (GR * GR)) | 0;
    if (x > 0) f(i - 1);
    if (x < GR - 1) f(i + 1);
    if (y > 0) f(i - GR);
    if (y < GR - 1) f(i + GR);
    if (z > 0) f(i - GR * GR);
    if (z < GR - 1) f(i + GR * GR);
  };
  // BFS the surface labels inward so interior splats get a colour
  while (qh < qt) {
    const i = queue[qh++];
    const l = label[i];
    nb(i, j => {
      if (!outside[j] && label[j] < 0) {
        label[j] = l;
        queue[qt++] = j;
      }
    });
  }

  const fill = [];
  const fillLabel = [];
  let seed = 3;
  const rnd = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  for (let z = 0; z < GR; z++)
    for (let y = 0; y < GR; y++)
      for (let x = 0; x < GR; x++) {
        const i = x + y * GR + z * GR * GR;
        if (outside[i] || surfOcc[i]) continue;
        // keep the fill a voxel under the surface
        let buried = true;
        nb(i, j => {
          if (outside[j]) buried = false;
        });
        if (buried) {
          fill.push(
            (x + 0.3 + rnd() * 0.4) / R2,
            (y + 0.3 + rnd() * 0.4) / R2,
            (z + 0.3 + rnd() * 0.4) / R2,
          );
          fillLabel.push(label[i]);
        }
      }
  const nFill = fill.length / 3;
  const n = nSurf + nFill;
  const rest = new Float32Array(n * 12); // rest covariance (6 + 2 pad) + colour
  const init = new Float32Array(n * 48); // matches Particle in sim.wgsl
  const iso = 0.3 * 0.3; // interior splats are small, round, opaque
  for (let i = 0; i < n; i++) {
    const o = i * 48;
    if (i < nSurf) {
      init.set(surf.subarray(i * 3, i * 3 + 3), o);
      const si = keep[i];
      const c = s.cov.subarray(si * 6, si * 6 + 6);
      const s2 = scale * scale;
      rest.set([c[0] * s2, c[1] * s2, c[2] * s2, c[3] * s2, c[4] * s2, c[5] * s2, 0, 0], i * 12);
      rest.set(s.col.subarray(si * 4, si * 4 + 4), i * 12 + 8);
    } else {
      const k = (i - nSurf) * 3;
      const l = fillLabel[i - nSurf];
      init[o] = fill[k];
      init[o + 1] = fill[k + 1];
      init[o + 2] = fill[k + 2];
      rest.set([iso, 0, 0, iso, 0, iso, 0, 0], i * 12);
      if (l >= 0) {
        const si = keep[l];
        rest.set(
          [s.col[si * 4] * 0.85, s.col[si * 4 + 1] * 0.85, s.col[si * 4 + 2] * 0.85, 1],
          i * 12 + 8,
        );
      } else rest.set([0.5, 0.5, 0.5, 1], i * 12 + 8);
    }
    init[o + 3] = 1; // mass
    init[o + 7] = 0.125; // volume, fixed up by the density pass
    init[o + 20] = 1;
    init[o + 25] = 1;
    init[o + 30] = 1; // F = I
    init[o + 44] = 1; // Jp
  }
  return { n, nSurf, nFill, init, rest };
}
