import numpy as np
import subprocess
import os
import re
import shutil
import multiprocessing
import time
from scipy.optimize import differential_evolution

multiprocessing.set_start_method("fork")

XFOIL_PATH = shutil.which("xfoil")
if XFOIL_PATH is None:
    raise RuntimeError("xfoil not found on PATH. Run: brew install xfoil")

TMPDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_xfoil_tmp")
os.makedirs(TMPDIR, exist_ok=True)

# ANSI colors
WHITE  = "\033[38;5;253m"
YELLOW = "\033[38;5;221m"
CYAN   = "\033[38;5;117m"
GREEN  = "\033[38;5;114m"
ORANGE = "\033[38;5;208m"
RED    = "\033[38;5;203m"
DIM    = "\033[38;5;240m"
MUTED  = "\033[38;5;245m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

MIN_LD_TO_CONFIRM = 30.0
CONFIRM_MARGIN    = 0.5


def fmt_time(s):
    return f"[{s:5.1f}s]"


def fmt_re_col(re, i, scores):
    re_label = f"{CYAN}Re={re//1000}k{RESET}"
    val = scores[i]  # scores[i] always maps to reynolds_range[i], nans included
    if np.isfinite(val):
        val_color = GREEN if val > 40 else WHITE
        return f"{re_label}{DIM}:{RESET}{val_color}{val:7.2f}{RESET}"
    return f"{re_label}{DIM}:{RESET}{RED}   fail{RESET}"


def print_sanity(re, score):
    if np.isfinite(score):
        print(f"  {MUTED}Re={re}:{RESET} {WHITE}{score:.4f}{RESET}")
    else:
        print(f"  {MUTED}Re={re}:{RESET} {RED}fail{RESET}")


def get_user_parameters():
    bar = f"{DIM}{'─'*60}{RESET}"
    print(f"\n{bar}")
    print(f"  {WHITE}{BOLD}xfoil naca airfoil optimizer{RESET}")
    print(f"{bar}\n")

    print(f"{MUTED}[1] reynolds numbers{RESET}  {DIM}default: 80000, 150000, 300000{RESET}")
    raw = input(f"    {DIM}>{RESET} ").strip()
    reynolds_range = [int(r.strip()) for r in raw.split(",")] if raw else [80000, 150000, 300000]
    print(f"    {DIM}using: {reynolds_range}{RESET}\n")

    print(f"{MUTED}[2] angle of attack sweep{RESET}")
    raw = input(f"    {DIM}start (default -5)  >{RESET} ").strip()
    aoa_start = float(raw) if raw else -5.0
    raw = input(f"    {DIM}end   (default  5)  >{RESET} ").strip()
    aoa_end = float(raw) if raw else 5.0
    raw = input(f"    {DIM}step  (default  2)  >{RESET} ").strip()
    aoa_step = float(raw) if raw else 2.0
    print(f"    {DIM}sweep: {aoa_start} to {aoa_end}, step {aoa_step}{RESET}\n")

    print(f"{MUTED}[3] naca bounds{RESET}  {DIM}m=max-camber  p=camber-pos  t=thickness{RESET}")
    raw = input(f"    {DIM}m min (default 0)  >{RESET} ").strip(); m_min = float(raw) if raw else 0.0
    raw = input(f"    {DIM}m max (default 9)  >{RESET} ").strip(); m_max = float(raw) if raw else 9.0
    raw = input(f"    {DIM}p min (default 1)  >{RESET} ").strip(); p_min = float(raw) if raw else 1.0
    raw = input(f"    {DIM}p max (default 9)  >{RESET} ").strip(); p_max = float(raw) if raw else 9.0
    raw = input(f"    {DIM}t min (default  6) >{RESET} ").strip(); t_min = float(raw) if raw else 6.0
    raw = input(f"    {DIM}t max (default 18) >{RESET} ").strip(); t_max = float(raw) if raw else 18.0
    print()

    print(f"{MUTED}[4] optimizer{RESET}")
    raw = input(f"    {DIM}max iterations (default 100) >{RESET} ").strip(); maxiter = int(raw) if raw else 100
    raw = input(f"    {DIM}population size (default  5) >{RESET} ").strip(); popsize = int(raw) if raw else 5
    raw = input(f"    {DIM}random seed     (default 42) >{RESET} ").strip(); seed    = int(raw) if raw else 42
    print()

    print(f"{MUTED}[5] l/d cap{RESET}  {DIM}values above this are treated as convergence artifacts{RESET}")
    raw = input(f"    {DIM}max l/d (default 60) >{RESET} ").strip()
    max_ld = float(raw) if raw else 60.0
    print(f"\n{bar}\n")

    return {
        "reynolds_range": reynolds_range,
        "aoa_start": aoa_start,
        "aoa_end": aoa_end,
        "aoa_step": aoa_step,
        "bounds": [(m_min, m_max), (p_min, p_max), (t_min, t_max)],
        "maxiter": maxiter,
        "popsize": popsize,
        "seed": seed,
        "max_ld": max_ld,
    }


def naca4_coordinates(m, p, t, n_points=100):
    m_f = m / 100.0
    p_f = p / 10.0
    t_f = t / 100.0

    x = np.linspace(0, 1, n_points)
    yt = 5 * t_f * (
        0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2
        + 0.2843 * x**3 - 0.1015 * x**4
    )

    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)

    if p_f > 0 and m_f > 0:
        fore = x < p_f
        aft = ~fore
        yc[fore] = (m_f / p_f**2) * (2 * p_f * x[fore] - x[fore]**2)
        dyc_dx[fore] = (2 * m_f / p_f**2) * (p_f - x[fore])
        yc[aft] = (m_f / (1 - p_f)**2) * ((1 - 2*p_f) + 2*p_f*x[aft] - x[aft]**2)
        dyc_dx[aft] = (2 * m_f / (1 - p_f)**2) * (p_f - x[aft])

    theta = np.arctan(dyc_dx)
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    x_coords = np.concatenate([xu[::-1], xl[1:]])
    y_coords = np.concatenate([yu[::-1], yl[1:]])
    return np.column_stack([x_coords, y_coords])


def write_coords_file(coords, filepath):
    with open(filepath, 'w') as f:
        f.write("AIRFOIL\n")
        for x, y in coords:
            f.write(f"  {x:.6f}  {y:.6f}\n")


def load_polar_file(filepath):
    rows = []
    with open(filepath, 'r') as f:
        for line in f:
            stripped = line.strip()
            if stripped and re.match(r'^-?\d', stripped):
                try:
                    rows.append([float(v) for v in stripped.split()])
                except ValueError:
                    continue
    return np.array(rows) if rows else None


def parse_xfoil_output(stdout):
    alphas, cls, cds = [], [], []
    polar_line = re.compile(
        r'^\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(\d+\.\d+)\s+(-?\d+\.\d+)'
    )
    in_polar = False
    for line in stdout.splitlines():
        if '------' in line:
            in_polar = True
            continue
        if in_polar:
            m = polar_line.match(line)
            if m:
                alphas.append(float(m.group(1)))
                cls.append(float(m.group(2)))
                cds.append(float(m.group(3)))
    return np.array(alphas), np.array(cls), np.array(cds)


def run_xfoil(coords, reynolds, aoa_start, aoa_end, aoa_step, max_ld):
    pid = os.getpid()
    coord_file = os.path.join(TMPDIR, f"airfoil_{pid}.dat")
    polar_file  = os.path.join(TMPDIR, f"polar_{pid}.dat")

    if os.path.exists(polar_file):
        os.remove(polar_file)

    write_coords_file(coords, coord_file)

    commands = f"""PLOP
G F

LOAD {coord_file}
PANE
OPER
VISC {reynolds}
ITER 100
PACC
{polar_file}

ASEQ {aoa_start} {aoa_end} {aoa_step}
PACC


QUIT
"""

    try:
        result = subprocess.run(
            [XFOIL_PATH], input=commands,
            capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        return np.nan
    except FileNotFoundError:
        raise RuntimeError("xfoil not found on PATH.")

    if os.path.exists(polar_file) and os.path.getsize(polar_file) > 0:
        data = load_polar_file(polar_file)
        if data is None or data.size == 0:
            return np.nan
        if data.ndim == 1:
            data = data[np.newaxis, :]
        cl, cd = data[:, 1], data[:, 2]
    else:
        _, cl, cd = parse_xfoil_output(result.stdout)
        if len(cl) == 0:
            return np.nan

    valid = np.isfinite(cl) & np.isfinite(cd) & (cd > 1e-6)
    if valid.sum() == 0:
        return np.nan

    ld = float(np.mean(cl[valid] / cd[valid]))
    return ld if ld <= max_ld else np.nan


def evaluate_ld(coords, reynolds_range, aoa_start, aoa_end, aoa_step, max_ld):
    scores = []
    for Re in reynolds_range:
        score = run_xfoil(coords, Re, aoa_start, aoa_end, aoa_step, max_ld)
        scores.append(score)  # keep nan in place — preserves index alignment with reynolds_range
    finite = [s for s in scores if np.isfinite(s)]
    avg_ld = np.mean(finite) if finite else float('nan')
    return scores, avg_ld


if __name__ == "__main__":
    cfg = get_user_parameters()

    reynolds_range = cfg["reynolds_range"]
    aoa_start      = cfg["aoa_start"]
    aoa_end        = cfg["aoa_end"]
    aoa_step       = cfg["aoa_step"]
    MAX_LD         = cfg["max_ld"]

    print(f"{DIM}sanity check{RESET}")
    test_coords = naca4_coordinates(4, 4, 12)
    for Re in reynolds_range:
        score = run_xfoil(test_coords, Re, aoa_start, aoa_end, aoa_step, MAX_LD)
        print_sanity(Re, score)
    print()

    start_time    = time.time()
    ld_results    = []
    verified_best = {'naca': None, 'ld': -np.inf, 'params': None}
    eval_count    = [0]  # list so closure can mutate it

    def objective(params):
        m, p, t = params
        coords  = naca4_coordinates(m, p, t)
        scores, avg_ld = evaluate_ld(coords, reynolds_range, aoa_start, aoa_end, aoa_step, MAX_LD)

        eval_count[0] += 1
        n       = eval_count[0]
        elapsed = time.time() - start_time
        naca    = f"{int(round(m))}{int(round(p))}{int(round(t)):02d}"

        re_cols   = "  ".join(fmt_re_col(Re, i, scores) for i, Re in enumerate(reynolds_range))
        avg_color = GREEN if np.isfinite(avg_ld) and avg_ld > 40 else WHITE
        print(f"  {DIM}#{n:<4} {fmt_time(elapsed)}{RESET}  {WHITE}{BOLD}NACA {naca}{RESET}  {re_cols}  {DIM}avg{RESET} {avg_color}{avg_ld:.4f}{RESET}")

        if np.isfinite(avg_ld) and avg_ld >= MIN_LD_TO_CONFIRM and avg_ld > verified_best['ld'] + CONFIRM_MARGIN:
            scores_2, ld_2 = evaluate_ld(coords, reynolds_range, aoa_start, aoa_end, aoa_step, MAX_LD)
            confirmed_ld   = np.mean([avg_ld, ld_2]) if np.isfinite(ld_2) else avg_ld

            if confirmed_ld > verified_best['ld']:
                verified_best['ld']     = confirmed_ld
                verified_best['naca']   = naca
                verified_best['params'] = (m, p, t)
                print(f"  {GREEN}confirmed{RESET}  {WHITE}{BOLD}NACA {naca}{RESET}  {DIM}run1{RESET} {YELLOW}{avg_ld:.4f}{RESET}  {DIM}run2{RESET} {YELLOW}{ld_2:.4f}{RESET}  {DIM}avg{RESET} {GREEN}{confirmed_ld:.4f}{RESET}")
            else:
                avg_ld = confirmed_ld
                print(f"  {RED}fluke{RESET}      {WHITE}{BOLD}NACA {naca}{RESET}  {DIM}run1{RESET} {YELLOW}{avg_ld:.4f}{RESET}  {DIM}run2{RESET} {YELLOW}{ld_2:.4f}{RESET}  {DIM}discarded{RESET}")

        ld_results.append((naca, avg_ld))
        return -avg_ld if np.isfinite(avg_ld) else 0.0

    def callback(xk, convergence):
        elapsed = time.time() - start_time
        if verified_best['naca'] is not None:
            print(f"\n  {DIM}best so far  [{elapsed:.1f}s]{RESET}  {WHITE}{BOLD}NACA {verified_best['naca']}{RESET}  {DIM}l/d{RESET} {GREEN}{verified_best['ld']:.4f}{RESET}  {DIM}convergence{RESET} {MUTED}{convergence:.6f}{RESET}\n")

    result = differential_evolution(
        objective,
        bounds=cfg["bounds"],
        maxiter=cfg["maxiter"],
        popsize=cfg["popsize"],
        seed=cfg["seed"],
        tol=1e-4,
        disp=True,
        workers=1,
        updating='immediate',
        callback=callback,
    )

    best_m, best_p, best_t = result.x
    best_naca   = f"{int(round(best_m))}{int(round(best_p))}{int(round(best_t)):02d}"
    best_avg_ld = -result.fun

    if verified_best['ld'] > best_avg_ld:
        best_naca   = verified_best['naca']
        best_avg_ld = verified_best['ld']
        best_m, best_p, best_t = verified_best['params']

    bar = f"{DIM}{'─'*60}{RESET}"
    print(f"\n{bar}")
    print(f"  {MUTED}airfoil{RESET}      {WHITE}{BOLD}NACA {best_naca}{RESET}")
    print(f"  {MUTED}params{RESET}       {DIM}m={best_m:.3f}  p={best_p:.3f}  t={best_t:.3f}{RESET}")
    print(f"  {MUTED}avg l/d{RESET}      {GREEN}{best_avg_ld:.4f}{RESET}")
    print(f"  {MUTED}evaluations{RESET}  {WHITE}{len(ld_results)}{RESET}")
    print(f"  {MUTED}time{RESET}         {WHITE}{time.time()-start_time:.1f}s{RESET}")
    print(f"{bar}\n")
