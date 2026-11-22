# Gaussian splats with MPM physics (WebGPU)

A 3D Gaussian splat object simulated with the material point method, all on the GPU in the browser. Pick jelly, sand, snow or liquid and the object drops onto the floor. Drag to orbit and click to poke it. You can load your own 3DGS `.ply` file.

```sh
git clone --recurse-submodules https://github.com/rcarty/rc-gsplat-mpm
cd rc-gsplat-mpm
python3 -m http.server        # then open http://localhost:8000
```

It needs a browser with WebGPU (recent Chrome, Edge, Safari or Firefox). There's no build step and there are no third-party dependencies.

## Layout

| Path                        | Contents                                                                                                                                                                       |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `index.html`, `src/main.js` | The page, GPU setup and frame                                                                                                                                                  |
| `src/scene.js`              | CPU side: the generated knot, the `.ply` loader and the interior fill                                                                                                          |
| `src/shaders/`              | The solver passes, the 3×3 SVD, splat projection and drawing                                                                                                                   |
| `kit/`                      | [rc-webgpu-kit](https://github.com/rcarty/rc-webgpu-kit) as a git submodule: the depth sort, MPM weights and grid, orbit camera and device helpers, shared with my other demos |

## Formatting and lint

```sh
npm install
npm run lint      # eslint, then prettier --check
npm run format    # prettier --write
```

The rules come from rc-webgpu-kit (`kit/prettier.config.js`, `kit/eslint.config.js`), so every repository that uses it is formatted the same way.

## How it works

**Simulation.** Every splat is an MLS-MPM particle on a 64³ grid, using APIC transfers, quadratic B-spline weights and fixed-point atomic particle-to-grid scatter. Each particle carries a deformation gradient F. A 3×3 SVD runs in WGSL every substep (Jacobi eigen-iterations on FᵀF, then U from F V Σ⁻¹ with Gram-Schmidt, keeping U and V as rotations).

| Material | Model                                                                         |
| -------- | ----------------------------------------------------------------------------- |
| Jelly    | Fixed-corotated elasticity                                                    |
| Sand     | Drucker-Prager plasticity, return mapping in Hencky strain (Klar et al. 2016) |
| Snow     | Fixed-corotated with stretch clamping and hardening (Stomakhin et al. 2013)   |
| Liquid   | Volume-only response (F reset to J^⅓ I) plus a viscous term                   |

The floor applies Coulomb friction on the grid.

**Interior fill.** Trained splats only cover the surface, so the object is voxelised at half-cell resolution. A voxel counts as inside when it can see the surface in at least five of the six axis directions, which copes with the holes and missing underside of a real scan. The interior voxels get small round splats, kept one voxel under the surface. A multi-source BFS spreads the nearest surface splat's colour inward, so interior splats match the surface above them.

**Deforming the splats.** Each splat's covariance follows the material: Σ = F Σ₀ Fᵀ (PhysGaussian, Xie et al., CVPR 2024).

**Rendering.** A compute pass projects each 3D covariance to a screen-space conic (EWA splatting) and writes a 16-bit depth key. The key-value radix sort from rc-webgpu-kit (four 4-bit passes, the same kernels as the [sorting demo](https://github.com/rcarty/rc-gpu-sort)) orders the splats back to front, and they're drawn as instanced, alpha-blended quads.

## Loading your own splats

Standard 3DGS `.ply` files (binary little-endian with `x y z`, `f_dc_*`, `opacity`, `scale_*` and `rot_*`). Files are converted from COLMAP's y-down convention, and trimmed to the 2nd–98th percentile bounds to drop floaters. Files over 150k splats (50k on phones) are sampled down, weighted by opacity times area, and the splats that are kept are grown so the surface stays covered. Every splat gets the same density, with a cap on its rest volume so stray splats can't blow up. Colour uses spherical-harmonic degree 0 only.

## Limits

- One material for the whole object
- Dense 64³ grid, so the object is about 32 cells across
- The interior fill copes with holes and a missing underside, but not with a scan that is mostly open
- Splats stay flat discs under liquid, because liquid keeps no shear in F
