"""
xfoil_optimise.py
------------------
Optimises NACA 4-digit OR 6-series laminar-flow airfoils for maximum L/D
using XFOIL + differential evolution.

  NACA 4-digit  e.g. 2412  — classic series, any speed
  NACA 6-series e.g. 66-212 — laminar-flow family (P-51 Mustang style)
    Format: 6X-YZZ  where
      X   = series modifier (3,4,5,6)   — controls laminar bucket width
      Y   = ideal-lift coefficient × 10  e.g. 2 → Cli = 0.2
      ZZ  = max thickness % chord        e.g. 12 → 12%

Requirements:
    pip install numpy scipy
    brew install xfoil   # macOS
    sudo apt install xfoil  # Linux

Usage:
    python xfoil_optimise.py
"""

import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np
from scipy.optimize import differential_evolution

multiprocessing.set_start_method("fork")


# ═══════════════════════════════════════════════════════════════════
# NACA 4-digit geometry
# ═══════════════════════════════════════════════════════════════════

def naca4_coordinates(m, p, t, n_points=100):
    m_f = m / 100.0
    p_f = p / 10.0
    t_f = t / 100.0

    x = np.linspace(0, 1, n_points)

    yt = 5 * t_f * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )

    yc     = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)

    if p_f > 0 and m_f > 0:
        fore = x < p_f
        aft  = ~fore
        yc[fore]     = (m_f / p_f**2) * (2*p_f*x[fore] - x[fore]**2)
        dyc_dx[fore] = (2*m_f / p_f**2) * (p_f - x[fore])
        yc[aft]      = (m_f / (1-p_f)**2) * ((1-2*p_f) + 2*p_f*x[aft] - x[aft]**2)
        dyc_dx[aft]  = (2*m_f / (1-p_f)**2) * (p_f - x[aft])

    theta = np.arctan(dyc_dx)
    xu = x - yt*np.sin(theta);  yu = yc + yt*np.cos(theta)
    xl = x + yt*np.sin(theta);  yl = yc - yt*np.cos(theta)

    return np.column_stack([
        np.concatenate([xu[::-1], xl[1:]]),
        np.concatenate([yu[::-1], yl[1:]]),
    ])


# ═══════════════════════════════════════════════════════════════════
# NACA 6-series geometry
# ═══════════════════════════════════════════════════════════════════
#
# Thickness distributions from Abbott & von Doenhoff, Theory of Wing
# Sections (Dover, 1959), Table 6.2 / NACA Report 824.
#
# Mean line: NACA a=1.0 uniform-loading mean line, scaled by Cli.
# This is the standard mean line for all 6-series airfoils.
# Closed-form from Abbott & von Doenhoff eq. 4.25.

# Polynomial thickness coefficients per sub-family (a0..a4):
#   y_t = (t/0.20) * (a0*sqrt(x) + a1*x + a2*x^2 + a3*x^3 + a4*x^4)
_6SERIES_THICK = {
    63: (0.29690, -0.12600, -0.35160,  0.28430, -0.10150),
    64: (0.29115, -0.12600, -0.35160,  0.28430, -0.10150),
    65: (0.28679, -0.12600, -0.35160,  0.28430, -0.10150),
    66: (0.26595, -0.15597, -0.36435,  0.28108, -0.10003),  # P-51 Mustang family
}


def _thickness_6series(series, t_frac, x):
    """6-series thickness half-distribution at positions x."""
    a0, a1, a2, a3, a4 = _6SERIES_THICK.get(series, _6SERIES_THICK[66])
    xc = np.clip(x, 0.0, 1.0)
    return (t_frac / 0.20) * (
        a0 * np.sqrt(xc)
        + a1 * xc
        + a2 * xc**2
        + a3 * xc**3
        + a4 * xc**4
    )


def _meanline_a1(x):
    """
    NACA a=1.0 mean line (uniform chordwise loading), normalised per unit Cli.
    Returns (yc/Cli, dyc_dx/Cli) at each x.
    Source: Abbott & von Doenhoff eq. 4.25 / NACA TN 1945.
    """
    eps = 1e-12
    x   = np.clip(x, eps, 1.0 - eps)

    yc_norm  = (1.0 / (2.0 * np.pi)) * (
        -x * np.log(x)
        + (1.0 - x) * np.log(1.0 - x)
        + 0.5 * (1.0 - x)
    )
    dyc_norm = (1.0 / (2.0 * np.pi)) * (
        -np.log(x) - 1.0
        - np.log(1.0 - x)
        + 1.0
        - 0.5
    )
    return yc_norm, dyc_norm


def naca6_coordinates(series, cli_x10, thickness_pct, n_points=150):
    """
    NACA 6-series airfoil coordinates.

    series        : 63, 64, 65, or 66
    cli_x10       : design lift coeff × 10  (integer 0–9)
    thickness_pct : max thickness % chord   (integer 8–21)
    """
    cli    = cli_x10 / 10.0
    t_frac = thickness_pct / 100.0

    # cosine spacing — denser near LE/TE
    beta = np.linspace(0, np.pi, n_points)
    x    = 0.5 * (1.0 - np.cos(beta))

    yt = _thickness_6series(series, t_frac, x)

    yc_n, dyc_n = _meanline_a1(x)
    yc     = cli * yc_n
    dyc_dx = cli * dyc_n

    theta = np.arctan(dyc_dx)
    xu = x - yt*np.sin(theta);  yu = yc + yt*np.cos(theta)
    xl = x + yt*np.sin(theta);  yl = yc - yt*np.cos(theta)

    # Normalise x to exactly [0,1] BEFORE closing TE
    x_all = np.concatenate([xu, xl])
    xmn, xmx = x_all.min(), x_all.max()
    xu = (xu - xmn) / (xmx - xmn)
    xl = (xl - xmn) / (xmx - xmn)

    # Close TE at exactly x=1, y=0 after normalisation
    xu = np.append(xu, 1.0);  yu = np.append(yu, 0.0)
    xl = np.append(xl, 1.0);  yl = np.append(yl, 0.0)

    # Confirm upper surface = the one with greater y at mid-chord
    mid = n_points // 2
    if yu[mid] < yl[mid]:
        xu, yu, xl, yl = xl, yl, xu, yu

    # XFOIL winding: upper surface TE->LE then lower surface LE->TE
    return np.column_stack([
        np.concatenate([xu[::-1], xl[1:]]),
        np.concatenate([yu[::-1], yl[1:]]),
    ])


# ── Unified helpers ───────────────────────────────────────────────

def make_coords(series_type, params, n_points):
    if series_type == "4digit":
        m, p, t = round(params[0]), round(params[1]), round(params[2])
        return naca4_coordinates(m, p, t, n_points), (m, p, t)
    else:
        s   = round(params[0])
        cli = round(params[1])
        t   = round(params[2])
        return naca6_coordinates(s, cli, t, n_points), (s, cli, t)


def label(series_type, key):
    if series_type == "4digit":
        m, p, t = key
        return f"NACA {m}{p}{t:02d}"
    else:
        s, cli, t = key
        return f"NACA {s}-{cli}{t:02d}"


# ═══════════════════════════════════════════════════════════════════
# XFOIL helpers
# ═══════════════════════════════════════════════════════════════════

def write_coords(coords, path):
    with open(path, "w") as f:
        f.write("AIRFOIL\n")
        for x, y in coords:
            f.write(f"{x:.6f} {y:.6f}\n")


def load_polar(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if s and re.match(r"^-?\d", s):
                try:
                    rows.append([float(v) for v in s.split()])
                except ValueError:
                    continue
    return np.array(rows) if rows else None


def run_xfoil(coords, reynolds, cfg):
    pid   = os.getpid()
    cfile = os.path.join(cfg.tmpdir, f"airfoil_{pid}.dat")
    pfile = os.path.join(cfg.tmpdir, f"polar_{pid}.dat")

    if os.path.exists(pfile):
        os.remove(pfile)

    write_coords(coords, cfile)
    a0, a1, as_ = cfg.aoa

    cmds = (
        f"PLOP\nG F\n\n"           # turn off graphics
        f"LOAD {cfile}\n"          # load airfoil file
        f"\n"                      # accept default airfoil name
        f"PPAR\nN 160\n\n"         # set panel count, blank to exit PPAR
        f"OPER\n"                  # enter OPER menu
        f"VISC {reynolds}\n"       # set Reynolds number
        f"VPAR\nN {cfg.ncrit}\n\n" # set Ncrit, blank to exit VPAR
        f"ITER {cfg.xfoil_iter}\n" # set iteration limit
        f"PACC\n{pfile}\n\n"       # start polar accumulation
        f"ASEQ {a0} {a1} {as_}\n"  # run AoA sweep
        f"PACC\n"                  # stop polar accumulation
        f"\nQUIT\n"                # exit OPER, quit
    )

    try:
        subprocess.run([cfg.xfoil_path], input=cmds,
                       capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return np.nan

    if not os.path.exists(pfile):
        return np.nan

    data = load_polar(pfile)
    if data is None or data.size == 0:
        return np.nan
    if data.ndim == 1:
        data = data[np.newaxis, :]

    cl, cd = data[:, 1], data[:, 2]
    valid  = np.isfinite(cl) & np.isfinite(cd) & (cd > 1e-6)
    if valid.sum() == 0:
        return np.nan

    ld = cl[valid] / cd[valid]
    ld = ld[ld < 300]          # drop only physically absurd values
    if ld.size == 0:
        return np.nan

    med = float(np.median(ld))
    return np.nan if med > 250 else med  # 6-series at high Re can reach 120-150


# ═══════════════════════════════════════════════════════════════════
# Optimiser state
# ═══════════════════════════════════════════════════════════════════

wing_history  = {}
best_hist_avg = -float("inf")
best_key_glob = None
start_time    = None
_cfg          = None


def _eval(coords, key):
    """Run all Reynolds numbers, return clipped mean L/D."""
    scores = []
    for Re in _cfg.reynolds:
        s = run_xfoil(coords, Re, _cfg)
        scores.append(float(s) if np.isfinite(s) else 0.0)
    return np.clip(np.array(scores, dtype=float), 0, 250), scores


def objective(params):
    global best_hist_avg, best_key_glob

    coords, key = make_coords(_cfg.series_type, params, _cfg.n_points)
    lbl         = label(_cfg.series_type, key)

    scores_arr, scores_raw = _eval(coords, key)
    avg_ld       = float(np.mean(scores_arr))

    wing_history.setdefault(key, []).append(avg_ld)
    hist_avg     = float(np.mean(wing_history[key]))
    tests        = len(wing_history[key])
    return_value = hist_avg          # snapshot before any retest

    if hist_avg > best_hist_avg:
        best_hist_avg = hist_avg
        best_key_glob = key
        elapsed = time.time() - start_time
        print(f"\n>>> NEW BEST [{elapsed:.1f}s]: {lbl} | "
              f"hist_avg_LD={best_hist_avg:.4f} | tests={tests}")

        print(f"    Retesting {lbl} to confirm...")
        retest_arr, retest_raw = _eval(coords, key)
        retest_avg  = float(np.mean(retest_arr))
        wing_history[key].append(retest_avg)
        hist_avg      = float(np.mean(wing_history[key]))
        tests         = len(wing_history[key])
        best_hist_avg = hist_avg

        for Re, sc in zip(_cfg.reynolds, retest_raw):
            tag = f"{sc:.4f}" if np.isfinite(sc) and sc > 0 else "FAILED"
            print(f"      Re={Re:<10}  L/D = {tag}")
        print(f"    Retest avg={retest_avg:.4f}  "
              f"-> hist_avg={hist_avg:.4f} over {tests} tests\n")

    elapsed = time.time() - start_time
    print(f"[{elapsed:6.1f}s] {lbl} | "
          f"Re scores={[round(s,2) for s in scores_raw]} | "
          f"hist_avg={hist_avg:.4f} tests={tests}")

    return -return_value


def callback(xk, convergence):
    best_key  = None
    best_hist = -float("inf")
    for k, h in wing_history.items():
        v = float(np.mean(h))
        if v > best_hist:
            best_hist = v
            best_key  = k
    if best_key is None:
        return
    elapsed = time.time() - start_time
    print(f"\n>>> CALLBACK BEST [{elapsed:.1f}s]: "
          f"{label(_cfg.series_type, best_key)} | "
          f"hist_avg_LD={best_hist:.4f} | convergence={convergence:.6f}\n")


# ═══════════════════════════════════════════════════════════════════
# Interactive prompts
# ═══════════════════════════════════════════════════════════════════

def ask(prompt, default, cast):
    try:
        raw = input(f"  {prompt} [{default}]: ").strip()
        return cast(raw) if raw else default
    except (EOFError, KeyboardInterrupt):
        print(); sys.exit(0)


def ask_list(prompt, default_list, cast):
    dstr = " ".join(str(x) for x in default_list)
    try:
        raw = input(f"  {prompt} [{dstr}]: ").strip()
        if not raw:
            return default_list
        return [cast(v) for v in raw.split()]
    except (EOFError, KeyboardInterrupt):
        print(); sys.exit(0)
    except ValueError as e:
        print(f"    ✗ {e}. Using default.")
        return default_list


def ask_range(prompt, lo, hi, cast):
    try:
        raw = input(f"  {prompt} [{lo} {hi}]: ").strip()
        if not raw:
            return [lo, hi]
        parts = raw.split()
        if len(parts) != 2:
            raise ValueError("enter two numbers")
        return [cast(parts[0]), cast(parts[1])]
    except (EOFError, KeyboardInterrupt):
        print(); sys.exit(0)
    except ValueError as e:
        print(f"    ✗ {e}. Using default.")
        return [lo, hi]


def ask_bool(prompt, default=True):
    yn = "Y/n" if default else "y/N"
    try:
        raw = input(f"  {prompt} [{yn}]: ").strip().lower()
        if not raw:
            return default
        return raw in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        print(); sys.exit(0)


class Config:
    pass


# ═══════════════════════════════════════════════════════════════════
# Setup wizard
# ═══════════════════════════════════════════════════════════════════

def prompt_config():
    cfg = Config()

    print("=" * 62)
    print("  XFOIL Airfoil Optimiser")
    print("  Press Enter to accept each default shown in [brackets].")
    print("=" * 62)

    # ── Family ──
    print("\n── Airfoil family ──")
    print("  1) NACA 4-digit  (e.g. 2412)   — classic, fast")
    print("  2) NACA 6-series (e.g. 66-212) — laminar flow, P-51 Mustang style")
    choice          = ask("Choose family", "1", str)
    cfg.series_type = "6series" if choice.strip() == "2" else "4digit"

    # ── Reynolds ──
    print("\n── Sweep parameters ──")
    if cfg.series_type == "6series":
        # Laminar bucket matters most at higher Re
        default_re = [500000, 1000000, 3000000]
        print("  (6-series laminar flow is effective above Re ~500k)")
    else:
        default_re = [80000, 150000, 300000]

    cfg.reynolds = ask_list("Reynolds numbers (space-separated)", default_re, int)
    aoa_s = ask("AoA start (°)", -4,  float)
    aoa_e = ask("AoA stop  (°)",  8,  float)
    aoa_st= ask("AoA step  (°)",  1,  float)
    cfg.aoa = [aoa_s, aoa_e, aoa_st]

    # ── Search bounds ──
    if cfg.series_type == "4digit":
        print("\n── NACA 4-digit search bounds ──")
        cfg.bounds_labels = [
            "Max-camber digit  (0–9)",
            "Camber-pos digit  (1–9)",
            "Thickness %       (6–18)",
        ]
        cfg.search_bounds = [
            ask_range("Max-camber digit range  (0–9)",   0,  9, int),
            ask_range("Camber-pos digit range  (1–9)",   1,  9, int),
            ask_range("Thickness % range       (6–18)",  6, 18, int),
        ]
    else:
        print("\n── NACA 6-series search bounds ──")
        print("  Series : 63=widest bucket  64  65  66=narrowest/thinnest (P-51)")
        print("  Cli×10 : 0=symmetric  2=0.2 (cruise)  4=0.4 (climb)")
        print("  Thickness: P-51 used 66-012 at root tapering to 66-209 at tip")
        cfg.bounds_labels = [
            "Series modifier   (63–66)",
            "Design Cl×10      (0–9)",
            "Thickness %       (8–18)",
        ]
        cfg.search_bounds = [
            ask_range("Series modifier range  (63–66)", 63, 66, int),
            ask_range("Design Cl×10 range     (0–9)",    0,  9, int),
            ask_range("Thickness % range      (8–18)",   8, 18, int),
        ]

    # ── Optimiser ──
    print("\n── Optimiser settings ──")
    cfg.maxiter   = ask("DE max iterations",              100,  int)
    cfg.popsize   = ask("DE population size factor",        5,  int)
    cfg.seed      = ask("Random seed",                     42,  int)
    cfg.tol       = ask("Convergence tolerance",         1e-4,  float)
    cfg.min_tests = ask("Min evaluations before trusting",  3,  int)

    # ── Misc ──
    print("\n── Misc ──")
    cfg.n_points   = ask("Airfoil surface point count",   100, int)
    cfg.xfoil_iter = ask("XFOIL viscous iteration limit", 200, int)
    cfg.ncrit      = ask("Ncrit (9=clean, 3=turbulent, 12=glider)",  9, float)
    cfg.tmpdir     = ask("Temp directory", "./_xfoil_tmp", str)

    print("  Path to xfoil binary (leave blank to auto-detect) [auto]: ",
          end="", flush=True)
    try:
        ov = input().strip()
    except (EOFError, KeyboardInterrupt):
        print(); sys.exit(0)

    if ov:
        if not os.path.isfile(ov):
            print(f"  ✗ xfoil not found at '{ov}'"); sys.exit(1)
        cfg.xfoil_path = ov
    else:
        found = shutil.which("xfoil")
        if found is None:
            print("  ✗ xfoil not found on PATH.\n"
                  "    macOS: brew install xfoil\n"
                  "    Linux: sudo apt install xfoil")
            sys.exit(1)
        cfg.xfoil_path = found

    cfg.do_sanity = ask_bool("Run sanity check before optimising?", True)
    return cfg


def print_config(cfg):
    a0, a1, as_ = cfg.aoa
    n_aoa  = len(np.arange(a0, a1 + as_*0.5, as_))
    family = "NACA 6-series (laminar)" if cfg.series_type == "6series" else "NACA 4-digit"
    print("\n" + "=" * 62)
    print("  Running with:")
    print("=" * 62)
    print(f"  Airfoil family     : {family}")
    print(f"  Reynolds numbers   : {cfg.reynolds}")
    print(f"  AoA sweep          : {a0}° → {a1}°, step {as_}°  ({n_aoa} pts)")
    for lbl, b in zip(cfg.bounds_labels, cfg.search_bounds):
        print(f"  {lbl:<32}: {b[0]} – {b[1]}")
    print(f"  DE maxiter/popsize : {cfg.maxiter} / {cfg.popsize}")
    print(f"  Ncrit              : {cfg.ncrit}")
    print(f"  XFOIL binary       : {cfg.xfoil_path}")
    print("=" * 62 + "\n")


def sanity_check(cfg):
    print("=== XFOIL SANITY CHECK ===")
    if cfg.series_type == "4digit":
        coords = naca4_coordinates(4, 4, 12, n_points=cfg.n_points)
        lbl    = "NACA 4412"
    else:
        coords = naca6_coordinates(66, 2, 12, n_points=cfg.n_points)
        lbl    = "NACA 66-212"
    print(f"  Test airfoil: {lbl}")
    ok = True
    for Re in cfg.reynolds:
        score = run_xfoil(coords, Re, cfg)
        if np.isfinite(score):
            print(f"  Re={Re:<10}  L/D = {score:.4f}  ✓")
        else:
            print(f"  Re={Re:<10}  FAILED  ✗"); ok = False
    print("==" * 14)
    return ok


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    global _cfg, start_time

    cfg = prompt_config()
    os.makedirs(cfg.tmpdir, exist_ok=True)
    _cfg = cfg

    print_config(cfg)

    if cfg.do_sanity:
        ok = sanity_check(cfg)
        if not ok:
            print("\n[WARN] Sanity check had failures — results may be unreliable.")
            if not ask_bool("Continue anyway?", default=False):
                sys.exit(0)
        print()

    start_time = time.time()

    differential_evolution(
        objective,
        bounds=[tuple(b) for b in cfg.search_bounds],
        maxiter=cfg.maxiter,
        popsize=cfg.popsize,
        seed=cfg.seed,
        tol=cfg.tol,
        disp=True,
        workers=1,
        updating="immediate",
        callback=callback,
    )

    # ── Find overall best ──
    best_key  = None
    best_hist = -float("inf")
    for k, h in wing_history.items():
        v = float(np.mean(h))
        if v > best_hist:
            best_hist = v
            best_key  = k

    # ── Final retest ──
    confirmed_ld = None
    if best_key:
        lbl = label(cfg.series_type, best_key)
        print(f"\n  Final retest of {lbl}...")
        coords, _ = make_coords(cfg.series_type,
                                [float(v) for v in best_key],
                                cfg.n_points)
        arr, raw = _eval(coords, best_key)
        confirmed_ld = float(np.mean(arr))
        wing_history[best_key].append(confirmed_ld)
        for Re, sc in zip(cfg.reynolds, raw):
            tag = f"{sc:.4f}" if sc > 0 else "FAILED"
            print(f"    Re={Re:<10}  L/D = {tag}")

    # ── Save best 6-series .dat ──
    if best_key and cfg.series_type == "6series":
        s, cli, t = best_key
        dat_name  = f"NACA_{s}-{cli}{t:02d}.dat"
        coords, _ = make_coords(cfg.series_type,
                                [float(v) for v in best_key],
                                cfg.n_points)
        write_coords(coords, dat_name)
        print(f"\n  Saved airfoil coordinates -> {dat_name}")

    # ── Summary ──
    elapsed = time.time() - start_time
    print("\n" + "=" * 62)
    if best_key:
        lbl       = label(cfg.series_type, best_key)
        n_tests   = len(wing_history[best_key])
        final_avg = float(np.mean(wing_history[best_key]))
        print(f"  Best airfoil       : {lbl}")
        print(f"  All-run avg L/D    : {final_avg:.4f}  ({n_tests} total tests)")
        if confirmed_ld is not None:
            print(f"  Final retest L/D   : {confirmed_ld:.4f}")
        print(f"  Total time         : {elapsed:.1f}s")
        print(f"  Candidates tested  : {len(wing_history)}")
    else:
        print("  No candidates were tested.")
    print("=" * 62)


if __name__ == "__main__":
    main()
