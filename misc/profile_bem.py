"""Standalone profiling harness for the BEM forward/backward solves.

Times ``DiffBEM.gradient()`` in isolation (no Mitsuba/CLIP) at the two
production mesh sizes, mirroring the real preprocessing path from
``acoustics_opt.py`` (box_mesher -> top-face detection -> face_subdivision ->
precompute). Reports a per-phase wall-clock breakdown and BiCGSTAB iteration
counts so we can see where the per-iteration time actually goes.

Usage:
    BEM_PROFILE=1 python misc/profile_bem.py            # single-band (2000 Hz)
    BEM_PROFILE=1 python misc/profile_bem.py --full     # full 7-band list

Note: BEM_PROFILE must be set in the environment; it is cached in the C++
extension on the first profiled call. We also set it here defensively before
importing acoustics3d.
"""

import os

# Enable profiling before importing the extension (env is cached on first use).
os.environ.setdefault("BEM_PROFILE", "1")

import argparse
import sys

import numpy as np

# make the repo root importable when run from anywhere
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import acoustics3d as ac3d
from pyoptim import mesher

# [diffbem] settings from configs/base.toml
DIFFBEM_CFG = dict(
    cluster_size=128,
    radius_factor=1.5,
    freq_bands=[1000, 1250, 1600, 1600, 2000, 2500, 3150],
    n_freqs=1,
    approx_ACA_tol=1e-5,
    Q_ACA_tol=1e-5,
    solver_tol=1e-5,
    src_pt=np.array([0.0, 100.0, 0.0]),
    listener_radius=50,
    listener_ds=5,
    recompute_matrices=False,
)

ESIZE = 0.02
SUBDIV_TOP = 1  # base.toml [dimensions] subdiv_top
PRIMARY_FREQ = 2000
WARMUP = int(os.environ.get("BEM_BENCH_WARMUP", "2"))
TIMED = int(os.environ.get("BEM_BENCH_TIMED", "5"))

MESHES = {
    "A (0.6m)": dict(w=0.6, b=0.6, h=0.15),
    "B (0.9m)": dict(w=0.9, b=0.9, h=0.15),
}


def build_diffbem(w, b, h):
    """Replicate acoustics_opt.py:180-202 (the real, box_mesher path)."""
    # NOTE: acoustics_opt.py currently calls generate_base_mesh on line 181; the
    # intended production path is box_mesher (line 180). We use box_mesher here.
    Ps, Es = mesher.box_mesher(ESIZE, w, b, h)

    # top-facing vertices == max y coordinate
    diff_pts_idx = np.where(np.isclose(Ps[:, 1], np.max(Ps[:, 1])))[0]

    # subdivide the top faces once
    if SUBDIV_TOP > 0:
        top_Es_idx = np.where(np.all(np.isin(Es, diff_pts_idx), axis=1))[0]
        Ps, Es = mesher.face_subdivision(Ps, Es, top_Es_idx, SUBDIV_TOP)
        diff_pts_idx = np.where(np.isclose(Ps[:, 1], np.max(Ps[:, 1])))[0]

    diffbem = ac3d.DiffBEM(**DIFFBEM_CFG)
    # precompute reorders Es and builds the cluster tree / direct-approx blocks
    Ps, Es = diffbem.precompute(Ps, Es, diff_pts_idx.tolist())
    diffbem.silent = True
    return diffbem, Ps, Es, diff_pts_idx


def profile_mesh(name, dims, full_bands=False):
    diffbem, Ps, Es, diff_pts_idx = build_diffbem(**dims)
    HH = len(diffbem.get_Hs())  # number of differentiable (top-face) points
    Ne = Es.shape[0]
    x = np.zeros(HH)  # flat heightfield -> base geometry

    if not full_bands:
        diffbem.set_band(PRIMARY_FREQ)
    band_desc = "7-band" if full_bands else f"{PRIMARY_FREQ}Hz"

    # warmup (JIT-free, but warms caches / thread pool), then reset and time
    for _ in range(WARMUP):
        diffbem.gradient(x)
    ac3d.profile_reset()

    val = None
    grad = None
    for _ in range(TIMED):
        val, grad = diffbem.gradient(x)

    gnorm = float(np.linalg.norm(grad)) if grad is not None else float("nan")
    print(f"\n{'=' * 70}")
    print(f"Mesh {name}: Ne={Ne} elements, HH={HH} diff points, band={band_desc}")
    print(f"  timed calls={TIMED}  last val={val:.6f}  |grad|={gnorm:.4e}")
    print(f"{'=' * 70}")
    print(ac3d.profile_summary())
    print(f"(divide total_s by {TIMED} timed calls for per-iteration cost)")


def _phase_means(labels):
    """Parse profile_summary() into {label: mean_ms}. Uses the total_s/calls
    columns (robust; the printed mean_ms/%tot columns can visually merge)."""
    out = {}
    for line in ac3d.profile_summary().splitlines():
        toks = line.split()
        if len(toks) >= 3 and toks[0] in labels:
            try:
                total_s, calls = float(toks[1]), float(toks[2])
                out[toks[0]] = total_s / calls * 1e3 if calls else float("nan")
            except ValueError:
                pass
    return out


def compare_mesh(name, dims):
    """Time the GPU vs CPU derivative phases on one mesh and report speedups."""
    if not getattr(ac3d.acoustics3d_impl, "cuda_available", False):
        print("CUDA backend not built (BEM_USE_CUDA=OFF); nothing to compare.", file=sys.stderr)
        return
    diffbem, Ps, Es, diff_pts_idx = build_diffbem(**dims)
    HH = len(diffbem.get_Hs())
    Ne = Es.shape[0]
    diffbem.set_band(PRIMARY_FREQ)
    x = 0.02 * np.sin(np.arange(HH))  # deterministic nonzero heights

    labels = ["fwd.aca_listener", "bwd.der_A", "bwd.der_B", "bwd.der_C",
              "fwd.bicgstab_solve", "bwd.bicgstab_solve", "fwd.hmat_assembly",
              "outer.freq_bands_total"]

    def run(use_gpu, fp32=False):
        diffbem.use_gpu = use_gpu
        diffbem.gpu_fp32 = fp32
        for _ in range(WARMUP):
            diffbem.gradient(x)
        ac3d.profile_reset()
        for _ in range(TIMED):
            val, grad = diffbem.gradient(x)
        return np.asarray(grad), _phase_means(labels)

    g_cpu, cpu = run(False)
    g_g64, g64 = run(True, fp32=False)
    g_g32, g32 = run(True, fp32=True)

    print(f"\n{'=' * 92}")
    print(f"Mesh {name}: Ne={Ne}, HH={HH}, band={PRIMARY_FREQ}Hz  (per-call mean over {TIMED})")
    print(f"{'=' * 92}")
    print(f"  {'phase':24s} {'CPU ms':>11} {'GPU64 ms':>11} {'GPU32 ms':>11} "
          f"{'64x':>7} {'32x':>7}")
    for lb in labels:
        c = cpu.get(lb, float('nan'))
        a = g64.get(lb, float('nan'))
        b = g32.get(lb, float('nan'))
        s64 = c / a if a else float('nan')
        s32 = c / b if b else float('nan')
        print(f"  {lb:24s} {c:11.1f} {a:11.1f} {b:11.1f} {s64:6.1f}x {s32:6.1f}x")
    n = max(np.linalg.norm(g_cpu), 1e-300)
    print(f"  grad rel-L2 vs CPU:  GPU64={np.linalg.norm(g_cpu-g_g64)/n:.2e}  "
          f"GPU32={np.linalg.norm(g_cpu-g_g32)/n:.2e}  "
          f"(multithreaded forward noise dominates; see fp32 accuracy test for clean numbers)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="profile the full 7-band list instead of a single 2000 Hz band")
    ap.add_argument("--compare", action="store_true",
                    help="time GPU vs CPU derivative phases (requires BEM_USE_CUDA build)")
    args = ap.parse_args()

    if not ac3d.profile_enabled():
        print("WARNING: profiling is DISABLED. Set BEM_PROFILE=1 before running "
              "(and before the extension is first imported).", file=sys.stderr)

    mesh_filter = os.environ.get("BEM_BENCH_MESH", "")  # substring filter on mesh name

    if args.compare:
        for name, dims in MESHES.items():
            if mesh_filter and mesh_filter not in name:
                continue
            compare_mesh(name, dims)
    else:
        for name, dims in MESHES.items():
            profile_mesh(name, dims, full_bands=args.full)


if __name__ == "__main__":
    main()
