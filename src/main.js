import {
  initGPU,
  buf,
  bind,
  computePipeline,
  fitCanvas,
  frameLoop,
  loadWGSL,
  mat4,
  pickDir,
  OrbitCamera,
  loadSort,
  createGrid,
  particleGroups,
} from '../kit/src/index.js';
import { G, FLOOR, makeKnot, parsePly, buildParticles } from './scene.js';

const $ = id => document.getElementById(id);
const canvas = $('view');
const status = t => {
  $('status').textContent = t;
};

const small = Math.min(innerWidth, innerHeight) < 700;
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
const S = 4 / G; // world units per cell
const SUBSTEPS = small ? 6 : 8;
const DT = 0.05;
const MAX_SPLATS = small ? 50000 : 150000;

const MATERIALS = [
  { name: 'Jelly', E: 220, nu: 0.3, friction: 0.3 },
  { name: 'Sand', E: 400, nu: 0.3, friction: 0.6, alpha: 0.386 },
  { name: 'Snow', E: 250, nu: 0.2, friction: 0.4, hardening: 10 },
  { name: 'Liquid', E: 0, nu: 0, la: 60, friction: 0.05, viscosity: 5 },
];

let device;
let context;
let sort;
let grid;
let pipes;
let scene = null;
let material = 0;
let resetPending = true;
let poke = null;

const cam = new OrbitCamera(canvas, {
  onTap: (sx, sy) => {
    poke = { sx, sy, frames: 2 };
  },
});

async function init() {
  let format;
  ({ device, context, format } = await initGPU({ canvas, alphaMode: 'opaque' }));
  device.lost.then(i => {
    status('GPU device lost: ' + i.message);
  });

  const shader = name => loadWGSL(new URL(`./shaders/${name}.wgsl`, import.meta.url));
  const names = ['p2g', 'p2g_mass', 'vol_init', 'grid', 'g2p', 'project', 'draw', 'floor'];
  const src = Object.fromEntries(await Promise.all(names.map(async n => [n, await shader(n)])));
  sort = await loadSort(device);
  grid = createGrid(device, [G, G, G]);

  const comp = code => computePipeline(device, code);
  const blend = {
    color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
    alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
  };
  const render = (code, target) => {
    const module = device.createShaderModule({ code });
    return device.createRenderPipeline({
      layout: 'auto',
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [target] },
    });
  };
  pipes = {
    p2g: comp(src.p2g),
    p2gMass: comp(src.p2g_mass),
    volInit: comp(src.vol_init),
    grid: comp(src.grid),
    g2p: comp(src.g2p),
    project: comp(src.project),
    draw: render(src.draw, { format, blend }),
    floor: render(src.floor, { format }),
  };
}

// upload a particle set and rebuild everything sized by it
function upload(built) {
  if (scene) scene.destroy();
  const u = GPUBufferUsage;
  const n = built.n;
  const B = (b, r) => bind(device, b, r);
  const particles = buf(device, n * 192, u.STORAGE | u.COPY_DST);
  const rest = buf(device, n * 48, u.STORAGE | u.COPY_DST);
  device.queue.writeBuffer(rest, 0, built.rest);
  const simU = buf(device, 112, u.UNIFORM | u.COPY_DST);
  const renU = buf(device, 176, u.UNIFORM | u.COPY_DST);
  const floorU = buf(device, 80, u.UNIFORM | u.COPY_DST);
  const outs = buf(device, n * 48, u.STORAGE);
  const keys = buf(device, n * 4, u.STORAGE);
  const tmpKeys = buf(device, n * 4, u.STORAGE);
  const values = buf(device, n * 4, u.STORAGE);
  const tmpValues = buf(device, n * 4, u.STORAGE);
  // sort back to front: 16-bit depth keys, splat index as value
  const sorter = sort.radix({ count: n, keys, tmpKeys, values, tmpValues, keyBits: 16 });
  const owned = [particles, rest, simU, renU, floorU, outs, keys, tmpKeys, values, tmpValues];

  scene = {
    n,
    nSurf: built.nSurf,
    nFill: built.nFill,
    init: built.init,
    particles,
    simU,
    renU,
    floorU,
    sorter,
    bg: {
      p2g: B(pipes.p2g, [particles, grid.cells, simU]),
      p2gMass: B(pipes.p2gMass, [particles, grid.cells, simU]),
      volInit: B(pipes.volInit, [particles, grid.cells, simU]),
      grid: B(pipes.grid, [grid.cells, grid.velocity, simU]),
      g2p: B(pipes.g2p, [particles, grid.velocity, simU]),
      project: B(pipes.project, [particles, rest, outs, keys, values, renU]),
      draw: B(pipes.draw, [outs, values, renU]),
      floor: B(pipes.floor, [floorU]),
    },
    destroy() {
      sorter.destroy();
      owned.forEach(b => b.destroy());
    },
  };
  resetPending = true;
}

const simAB = new ArrayBuffer(112);
const simI = new Int32Array(simAB);
const simF = new Float32Array(simAB);
const simUi = new Uint32Array(simAB);
const renF = new Float32Array(44);
const floorF = new Float32Array(20);
let owed = 0; // substeps not yet run

function writeSim(c) {
  const m = MATERIALS[material];
  const mu = m.E / (2 * (1 + m.nu));
  const la = m.la !== undefined ? m.la : (m.E * m.nu) / ((1 + m.nu) * (1 - 2 * m.nu));
  simI[0] = G;
  simI[1] = G;
  simI[2] = G;
  simUi[3] = scene.n;
  simF[4] = DT;
  simF[5] = 0.1;
  simF[6] = mu;
  simF[7] = la;
  simUi[8] = material;
  simF[9] = m.hardening || 0;
  simF[10] = m.alpha || 0;
  simF[11] = FLOOR;
  simF[12] = m.friction;
  simF[13] = 0;
  simF[14] = m.viscosity || 0;
  if (poke) {
    const dir = pickDir(c, poke.sx, poke.sy, c.fovy, c.aspect);
    const o = [c.eye[0] / S + G / 2, c.eye[1] / S + FLOOR, c.eye[2] / S + G / 2]; // eye in grid space
    simF[13] = 6;
    simF.set([o[0], o[1], o[2], 0], 16);
    simF.set([dir[0], dir[1], dir[2], 0], 20);
    simF.set([dir[0] * 1.2, dir[1] * 1.2 + 0.9, dir[2] * 1.2, 0], 24);
    if (--poke.frames <= 0) poke = null;
  }
  device.queue.writeBuffer(scene.simU, 0, simAB);
}

function encodeReset(enc) {
  device.queue.writeBuffer(scene.particles, 0, scene.init);
  enc.clearBuffer(grid.cells);
  const cp = enc.beginComputePass();
  const g = particleGroups(scene.n);
  cp.setPipeline(pipes.p2gMass);
  cp.setBindGroup(0, scene.bg.p2gMass);
  cp.dispatchWorkgroups(g);
  cp.setPipeline(pipes.volInit);
  cp.setBindGroup(0, scene.bg.volInit);
  cp.dispatchWorkgroups(g);
  cp.end();
}

function frame(now, dt) {
  fitCanvas(canvas, small ? 1.5 : 1.75);
  const W = canvas.width;
  const H = canvas.height;
  cam.update(dt, reduce);
  const c = cam.matrices(W / H);

  writeSim(c);
  renF.set(c.view, 0);
  renF.set(c.proj, 16);
  renF[32] = W;
  renF[33] = H;
  renF[34] = (c.proj[0] * W) / 2;
  renF[35] = (c.proj[5] * H) / 2;
  renF.set([S, G / 2, FLOOR, G / 2], 36);
  renF.set([c.near, c.far, scene.n, 0], 40);
  device.queue.writeBuffer(scene.renU, 0, renF);
  floorF.set(mat4.inverse(mat4.mul(c.proj, c.view)), 0);
  floorF.set([W, H, 0, 0], 16);
  device.queue.writeBuffer(scene.floorU, 0, floorF);

  const enc = device.createCommandEncoder();
  if (resetPending) {
    encodeReset(enc);
    resetPending = false;
  }
  const gP = particleGroups(scene.n);
  // SUBSTEPS per 60th of a second whatever the refresh rate, and no catching up after a slow frame
  owed = Math.min(owed + dt * 60 * SUBSTEPS, 2 * SUBSTEPS);
  const steps = Math.floor(owed);
  owed -= steps;
  for (let s = 0; s < steps; s++) {
    enc.clearBuffer(grid.cells);
    const cp = enc.beginComputePass();
    cp.setPipeline(pipes.p2g);
    cp.setBindGroup(0, scene.bg.p2g);
    cp.dispatchWorkgroups(gP);
    cp.setPipeline(pipes.grid);
    cp.setBindGroup(0, scene.bg.grid);
    cp.dispatchWorkgroups(grid.groups);
    cp.setPipeline(pipes.g2p);
    cp.setBindGroup(0, scene.bg.g2p);
    cp.dispatchWorkgroups(gP);
    cp.end();
  }
  const cp = enc.beginComputePass();
  cp.setPipeline(pipes.project);
  cp.setBindGroup(0, scene.bg.project);
  cp.dispatchWorkgroups(gP);
  scene.sorter.encode(cp);
  cp.end();

  const target = context.getCurrentTexture().createView();
  // the floor pass paints every pixel, so the clear colour never shows
  const rp = enc.beginRenderPass({
    colorAttachments: [{ view: target, loadOp: 'clear', storeOp: 'store' }],
  });
  rp.setPipeline(pipes.floor);
  rp.setBindGroup(0, scene.bg.floor);
  rp.draw(3);
  rp.setPipeline(pipes.draw);
  rp.setBindGroup(0, scene.bg.draw);
  rp.draw(6, scene.n);
  rp.end();
  device.queue.submit([enc.finish()]);
}

function setMaterial(m) {
  material = m;
  document
    .querySelectorAll('[data-mat]')
    .forEach(b => b.setAttribute('aria-pressed', +b.dataset.mat === m));
  resetPending = true;
}
document
  .querySelectorAll('[data-mat]')
  .forEach(b => b.addEventListener('click', () => setMaterial(+b.dataset.mat)));
$('drop').addEventListener('click', () => {
  resetPending = true;
});

let knotBuilt = null;
function describe(label) {
  status(
    `${label} · ${scene.nSurf.toLocaleString()} surface splats + ${scene.nFill.toLocaleString()} interior splats · ${G}³ grid`,
  );
}
function loadKnot() {
  if (!knotBuilt) knotBuilt = buildParticles(makeKnot(small ? 22000 : 65000), 12);
  upload(knotBuilt);
  cam.target = [0, 1.2, 0];
  describe('Marble knot (generated)');
  $('knot').hidden = true;
}
$('knot').addEventListener('click', loadKnot);
$('ply').addEventListener('change', async e => {
  const file = e.target.files[0];
  if (!file) return;
  status(`Reading ${file.name}…`);
  try {
    const s = parsePly(await file.arrayBuffer(), MAX_SPLATS);
    status(`Filling the inside of ${file.name}…`);
    await new Promise(r => setTimeout(r, 20));
    upload(buildParticles(s, 8));
    cam.target = [0, 1.0, 0];
    describe(
      `${file.name}${s.total > s.n ? ` (${s.n.toLocaleString()} of ${s.total.toLocaleString()} splats)` : ''}`,
    );
    $('knot').hidden = false;
  } catch (err) {
    status('Could not load that file: ' + err.message);
  }
  e.target.value = '';
});

init()
  .then(() => {
    loadKnot();
    frameLoop(device, canvas, frame);
  })
  .catch(err => {
    console.warn(err);
    document.body.classList.add('no-gpu');
    // the note in the page says there's no WebGPU. anything else gets its own message
    if (navigator.gpu)
      document.querySelector('.nogpu').textContent = `Couldn't start: ${err.message}`;
    document
      .querySelectorAll('.controls button, .controls input')
      .forEach(b => (b.disabled = true));
  });
