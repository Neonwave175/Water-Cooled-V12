import numpy as np
import subprocess
import os
import re
import shutil
import multiprocessing
import time
from scipy.optimize import differential_evolution

multiprocessing.set_start_method("fork")

# Reynolds numbers for UAV speeds
reynolds_range = [80000, 150000, 300000]

# AoA sweep
aoa_start, aoa_end, aoa_step = -5, 5, 2

XFOIL_PATH = shutil.which("xfoil")
if XFOIL_PATH is None:
    raise RuntimeError("xfoil not found on PATH. Run: brew install xfoil")

TMPDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_xfoil_tmp")
os.makedirs(TMPDIR, exist_ok=True)

start_time = time.time()


# history of all wing tests
wing_history = {}

# track global best result
best_hist_avg = -float("inf")
best_key = None


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
            f.write(f"{x:.6f} {y:.6f}\n")


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


def run_xfoil(coords, reynolds):

    pid = os.getpid()

    coord_file = os.path.join(TMPDIR, f"airfoil_{pid}.dat")
    polar_file = os.path.join(TMPDIR, f"polar_{pid}.dat")

    if os.path.exists(polar_file):
        os.remove(polar_file)

    write_coords_file(coords, coord_file)

    commands = f"""
PLOP
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
        subprocess.run(
            [XFOIL_PATH],
            input=commands,
            capture_output=True,
            text=True,
            timeout=30
        )

    except subprocess.TimeoutExpired:
        return np.nan

    if not os.path.exists(polar_file):
        return np.nan

    data = load_polar_file(polar_file)

    if data is None or data.size == 0:
        return np.nan

    if data.ndim == 1:
        data = data[np.newaxis, :]

    cl = data[:,1]
    cd = data[:,2]

    valid = np.isfinite(cl) & np.isfinite(cd) & (cd > 1e-6)

    if valid.sum() == 0:
        return np.nan

    ld = cl[valid] / cd[valid]

    ld = ld[ld < 200]

    if ld.size == 0:
        return np.nan

    ld_med = float(np.median(ld))

    if ld_med > 90:
        return np.nan

    return ld_med


def objective(params):

    m, p, t = params
    # round parameters from SciPy so we test discrete NACA values
    m = round(m)
    p = round(p)
    t = round(t)

    coords = naca4_coordinates(m, p, t)

    scores = []

    for Re in reynolds_range:
        score = run_xfoil(coords, Re)

        if np.isfinite(score):
            scores.append(score)
        else:
            scores.append(0.0)

    scores_arr = np.array(scores, dtype=float)

    scores_arr = np.clip(scores_arr, 0, 80)

    # use mean across Reynolds numbers
    avg_ld = float(np.mean(scores_arr)) if scores_arr.size else 0.0

    key = (round(m,3), round(p,3), round(t,3))

    if key not in wing_history:
        wing_history[key] = []

    wing_history[key].append(avg_ld)

    hist_avg = np.mean(wing_history[key])
    tests = len(wing_history[key])

    global best_hist_avg, best_key

    # update BEST SO FAR whenever historical average improves
    if hist_avg > best_hist_avg:
        best_hist_avg = hist_avg
        best_key = key

        elapsed_best = time.time() - start_time
        best_m, best_p, best_t = key
        best_naca = f"{int(round(best_m))}{int(round(best_p))}{int(round(best_t)):02d}"

        print(
            f"\n>>> BEST SO FAR [{elapsed_best:.1f}s]: "
            f"NACA {best_naca} | hist_avg_LD={best_hist_avg:.4f} "
            f"| tests={tests}\n"
        )

    elapsed = time.time() - start_time
    naca = f"{int(round(m))}{int(round(p))}{int(round(t)):02d}"

    print(
        f"[{elapsed:6.1f}s] NACA {naca} | "
        f"Re scores={scores} | hist_avg={hist_avg:.4f} tests={tests}"
    )

    return -hist_avg


def callback(xk, convergence):

    best_key = None
    best_hist = -float("inf")

    for k, history in wing_history.items():

        # require ≥3 tests before trusting
        if len(history) < 3:
            continue

        hist = float(np.mean(history))

        if hist > best_hist:
            best_hist = hist
            best_key = k

    if best_key is None:
        return

    m, p, t = best_key
    naca = f"{int(round(m))}{int(round(p))}{int(round(t)):02d}"

    elapsed = time.time() - start_time

    print(
        f"\n>>> BEST SO FAR [{elapsed:.1f}s]: "
        f"NACA {naca} | hist_avg_LD={best_hist:.4f} "
        f"| convergence={convergence:.6f}\n"
    )


if __name__ == "__main__":

    print("=== XFOIL SANITY CHECK ===")

    test_coords = naca4_coordinates(4,4,12)

    for Re in reynolds_range:
        score = run_xfoil(test_coords, Re)

        if np.isfinite(score):
            print(f"Re={Re} L/D = {score:.4f}")
        else:
            print(f"Re={Re} FAILED")

    print("==========================\n")

    start_time = time.time()

    result = differential_evolution(
        objective,
        bounds=[
            (0,9),
            (1,9),
            (6,18),
        ],
        maxiter=100,
        popsize=5,
        seed=42,
        tol=1e-4,
        disp=True,
        workers=1,
        updating='immediate',
        callback=callback
    )

    # find final best
    best_key = None
    best_hist = -float("inf")

    for k, history in wing_history.items():
        if len(history) < 3:
            continue

        hist = float(np.mean(history))

        if hist > best_hist:
            best_hist = hist
            best_key = k

    if best_key is not None:

        m, p, t = best_key
        naca = f"{int(round(m))}{int(round(p))}{int(round(t)):02d}"

        print("\n============================================================")
        print(f"Best airfoil (historical): NACA {naca}")
        print(f"Historical Avg L/D: {best_hist:.4f}")
        print(f"Total time: {time.time()-start_time:.1f}s")
        print("============================================================")