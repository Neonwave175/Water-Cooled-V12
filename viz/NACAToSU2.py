"""
generate_mesh_naca.py
----------------------
Generates a C-type unstructured mesh around a NACA 4-digit OR 6-series airfoil,
visualises the profile + mesh topology, and exports a .su2 file.

Usage:
    python generate_mesh_naca.py                  # prompts for designation
    python generate_mesh_naca.py 2412             # NACA 4-digit
    python generate_mesh_naca.py 66-212           # NACA 6-series
    python generate_mesh_naca.py 64-811 --radius 30 --le-size 0.004
    python generate_mesh_naca.py --help

Arguments:
    designation       4-digit (e.g. 2412) or 6-series (e.g. 66-212) NACA code
    --output FILE     output .su2 filename  (default: mesh_NACA<XXX>.su2)
    --radius R        far-field domain radius in chord lengths (default: 20)
    --le-size S       leading-edge mesh size (default: 0.005)
    --wake-size S     wake region mesh size  (default: 0.05)
    --far-size S      far-field mesh size    (default: 2.0)
    --n-points N      airfoil surface points (default: 150)
    --no-viz          skip visualisation
    --no-mesh         skip mesh generation (viz only)
"""

import argparse
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


# ═══════════════════════════════════════════════════════════════════
# NACA 4-digit geometry
# ═══════════════════════════════════════════════════════════════════

def naca4(designation: str, n_points: int = 200):
    if len(designation) != 4 or not designation.isdigit():
        raise ValueError(f"'{designation}' is not a valid 4-digit NACA code.")

    m = int(designation[0]) / 100.0
    p = int(designation[1]) / 10.0
    t = int(designation[2:]) / 100.0

    beta = np.linspace(0, np.pi, n_points)
    x = 0.5 * (1 - np.cos(beta))

    yt = (t / 0.2) * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )

    if p > 0:
        yc = np.where(
            x < p,
            (m / p**2) * (2 * p * x - x**2),
            (m / (1 - p)**2) * (1 - 2 * p + 2 * p * x - x**2),
        )
        dyc_dx = np.where(
            x < p,
            (2 * m / p**2) * (p - x),
            (2 * m / (1 - p)**2) * (p - x),
        )
    else:
        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)

    theta = np.arctan(dyc_dx)
    xu = x  - yt * np.sin(theta);  yu = yc + yt * np.cos(theta)
    xl = x  + yt * np.sin(theta);  yl = yc - yt * np.cos(theta)

    return xu, yu, xl, yl, x, yc


# ═══════════════════════════════════════════════════════════════════
# NACA 6-series geometry
# ═══════════════════════════════════════════════════════════════════

_6SERIES_THICK = {
    63: (0.29690, -0.12600, -0.35160,  0.28430, -0.10150),
    64: (0.29115, -0.12600, -0.35160,  0.28430, -0.10150),
    65: (0.28679, -0.12600, -0.35160,  0.28430, -0.10150),
    66: (0.26595, -0.15597, -0.36435,  0.28108, -0.10003),
}


def _thickness_6(series, t_frac, x):
    a0, a1, a2, a3, a4 = _6SERIES_THICK[series]
    xc = np.clip(x, 0.0, 1.0)
    return (t_frac / 0.20) * (
        a0 * np.sqrt(xc) + a1 * xc + a2 * xc**2 + a3 * xc**3 + a4 * xc**4
    )


def _meanline_a1(x):
    eps = 1e-12
    x = np.clip(x, eps, 1.0 - eps)
    yc  = (1 / (2 * np.pi)) * (-x * np.log(x) + (1 - x) * np.log(1 - x) + 0.5 * (1 - x))
    dyc = (1 / (2 * np.pi)) * (-np.log(x) - 1 - np.log(1 - x) + 1 - 0.5)
    return yc, dyc


def _fix_te_overlap(xu, yu, xl, yl):
    x_check = np.linspace(0.90, 1.0, 500)
    yu_check = np.interp(x_check, xu, yu)
    yl_check = np.interp(x_check, xl, yl)

    gap = yu_check - yl_check
    cross_idx = np.where(gap < 0)[0]

    if cross_idx.size == 0:
        xu[-1] = xl[-1] = 1.0
        yu[-1] = yl[-1] = 0.0
        return xu, yu, xl, yl

    x_cut = x_check[cross_idx[0]]

    xu = xu[xu <= x_cut]
    yu = yu[:len(xu)]
    xl = xl[xl <= x_cut]
    yl = yl[:len(xl)]

    xu = np.append(xu, 1.0);  yu = np.append(yu, 0.0)
    xl = np.append(xl, 1.0);  yl = np.append(yl, 0.0)

    return xu, yu, xl, yl


def naca6(series: int, cli_x10: int, thickness_pct: int, n_points: int = 150):
    cli    = cli_x10 / 10.0
    t_frac = thickness_pct / 100.0

    beta = np.linspace(0, np.pi, n_points)
    x    = 0.5 * (1.0 - np.cos(beta))

    yt = _thickness_6(series, t_frac, x)
    yc_n, dyc_n = _meanline_a1(x)
    yc     = cli * yc_n
    dyc_dx = cli * dyc_n

    theta = np.arctan(dyc_dx)
    xu = x - yt * np.sin(theta);  yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta);  yl = yc - yt * np.cos(theta)

    x_all = np.concatenate([xu, xl])
    xmn, xmx = x_all.min(), x_all.max()
    xu = (xu - xmn) / (xmx - xmn)
    xl = (xl - xmn) / (xmx - xmn)

    mid = n_points // 2
    if yu[mid] < yl[mid]:
        xu, yu, xl, yl = xl, yl, xu, yu

    xu, yu, xl, yl = _fix_te_overlap(xu, yu, xl, yl)

    xc_line = np.linspace(0, 1, 500)
    yc_line = cli * _meanline_a1(xc_line)[0]

    return xu, yu, xl, yl, xc_line, yc_line


# ═══════════════════════════════════════════════════════════════════
# Parsing
# ═══════════════════════════════════════════════════════════════════

def parse_designation(s: str):
    s = s.strip().replace(" ", "")

    m6 = re.fullmatch(r"(6[3-6])-?([0-9])([0-9]{2})", s)
    if m6:
        series  = int(m6.group(1))
        cli_x10 = int(m6.group(2))
        thick   = int(m6.group(3))
        return "6series", (series, cli_x10, thick)

    if re.fullmatch(r"[0-9]{4}", s):
        return "4digit", s

    raise ValueError(
        f"Cannot parse '{s}'.\n"
        "  4-digit : e.g. 2412\n"
        "  6-series: e.g. 66-212 or 64-811"
    )


def get_surfaces(series_type, key, n_points):
    if series_type == "4digit":
        xu, yu, xl, yl, xc, yc = naca4(key, n_points)
        return xu, yu, xl, yl, xc, yc
    else:
        series, cli_x10, thick = key
        return naca6(series, cli_x10, thick, n_points)


def designation_label(series_type, key):
    if series_type == "4digit":
        return f"NACA {key}"
    else:
        s, c, t = key
        return f"NACA {s}-{c}{t:02d}"


def airfoil_properties(series_type, key):
    if series_type == "4digit":
        m = int(key[0]) / 100.0
        p = int(key[1]) / 10.0
        t = int(key[2:]) / 100.0
        return {
            "Series":          "NACA 4-digit",
            "Max camber":      f"{m*100:.0f}% chord",
            "Max camber pos.": f"{p*100:.0f}% chord",
            "Max thickness":   f"{t*100:.0f}% chord",
            "Profile type":    "Symmetric" if m == 0 else "Cambered",
        }
    else:
        s, c, t = key
        names = {63: "wide bucket", 64: "wide, good off-design",
                 65: "balanced", 66: "narrow bucket (P-51)"}
        return {
            "Series":          f"NACA {s}-series  ({names.get(s,'')})",
            "Design Cl":       f"{c/10:.1f}",
            "Max thickness":   f"{t}% chord",
            "Mean line":       "a=1.0 (uniform loading)",
        }


# ═══════════════════════════════════════════════════════════════════
# Visualisation  —  dark theme
# ═══════════════════════════════════════════════════════════════════

# ── dark palette ──────────────────────────────────────────────────
BG          = "#0d1117"   # near-black background (github dark)
BG_PANEL    = "#161b22"   # slightly lighter panel bg
GRID_COL    = "#21262d"   # subtle grid lines
BORDER_COL  = "#30363d"   # axis spines / tick marks
TEXT_COL    = "#e6edf3"   # primary text
TEXT_MUTED  = "#8b949e"   # secondary / annotation text

C_UPPER     = "#58a6ff"   # bright blue  — upper surface
C_LOWER     = "#ff7b72"   # soft red     — lower surface
C_CAMBER    = "#3fb950"   # green        — camber line
C_CHORD     = "#484f58"   # dark gray    — chord line
C_WAKE      = "#e3b341"   # amber        — wake region
C_DOMAIN    = "#56d364"   # bright green — far-field boundary
C_AF_FILL   = "#1f6feb"   # deep blue    — airfoil fill in domain panel
C_ARROW     = "#8b949e"   # muted gray   — inflow arrow


def _style_ax(ax):
    """Apply dark theme to a matplotlib axes."""
    ax.set_facecolor(BG_PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER_COL)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    ax.xaxis.label.set_color(TEXT_MUTED)
    ax.yaxis.label.set_color(TEXT_MUTED)
    ax.title.set_color(TEXT_COL)
    ax.grid(True, color=GRID_COL, linewidth=0.5, linestyle="--", alpha=0.6)


def visualise(series_type, key, n_points,
              domain_radius, le_size, wake_size, far_size):

    xu, yu, xl, yl, xc, yc = get_surfaces(series_type, key, n_points)
    props = airfoil_properties(series_type, key)
    lbl   = designation_label(series_type, key)
    slug  = lbl.replace("NACA ", "NACA").replace(" ", "").replace("-", "-")

    plt.rcParams.update({
        "text.color":        TEXT_COL,
        "axes.labelcolor":   TEXT_MUTED,
        "xtick.color":       TEXT_MUTED,
        "ytick.color":       TEXT_MUTED,
        "legend.facecolor":  BG_PANEL,
        "legend.edgecolor":  BORDER_COL,
        "legend.labelcolor": TEXT_COL,
    })

    fig = plt.figure(figsize=(14, 6), facecolor=BG)
    fig.suptitle(f"{lbl}  ·  Airfoil & C-mesh layout",
                 fontsize=14, fontweight="bold", y=0.97, color=TEXT_COL)

    gs = GridSpec(1, 2, figure=fig, wspace=0.35,
                  left=0.07, right=0.97, top=0.88, bottom=0.10)

    # ── Panel 1: airfoil cross-section ──────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    _style_ax(ax1)
    ax1.set_aspect("equal")

    ax1.fill_between(xu, yu, 0, alpha=0.12, color=C_UPPER)
    ax1.fill_between(xl, yl, 0, alpha=0.12, color=C_LOWER)
    ax1.plot([0, 1], [0, 0], color=C_CHORD, lw=0.8, linestyle="--", zorder=1)
    ax1.plot(xc, yc, color=C_CAMBER, lw=1.0, linestyle="--",
             label="Camber line", zorder=2)
    ax1.plot(xu, yu, color=C_UPPER, lw=2.0, label="Upper surface", zorder=3)
    ax1.plot(xl, yl, color=C_LOWER, lw=2.0, label="Lower surface", zorder=3)
    ax1.plot([xu[-1], xl[-1]], [yu[-1], yl[-1]], color=BORDER_COL, lw=1.2, zorder=3)

    # max thickness annotation
    yu_on_xl = np.interp(xl, xu, yu)
    thick_dist = yu_on_xl - yl
    idx_t = np.argmax(thick_dist)
    xt = xl[idx_t]
    yt_top = np.interp(xt, xu, yu)
    yt_bot = yl[idx_t]
    ax1.annotate("", xy=(xt, yt_top), xytext=(xt, yt_bot),
                 arrowprops=dict(arrowstyle="<->", color=TEXT_MUTED, lw=0.8))
    t_label = (f"{key[2]}%c" if series_type == "6series"
               else f"{key[2:]}%c")
    ax1.text(xt + 0.015, (yt_top + yt_bot) / 2,
             f"t = {t_label}", fontsize=8, color=TEXT_MUTED, va="center")

    # max camber annotation
    if np.max(np.abs(yc)) > 0.001:
        idx_c = np.argmax(np.abs(yc))
        ax1.annotate("", xy=(xc[idx_c], yc[idx_c]), xytext=(xc[idx_c], 0),
                     arrowprops=dict(arrowstyle="<->", color=C_CAMBER, lw=0.8))
        ax1.text(xc[idx_c] + 0.015, yc[idx_c] / 2,
                 f"m = {yc[idx_c]*100:.1f}%c",
                 fontsize=8, color=C_CAMBER, va="center")

    ax1.scatter([0, 1], [0, 0], s=30, color=TEXT_MUTED, zorder=5)
    ax1.text(-0.015, 0, "LE", ha="right", va="center", fontsize=8, color=TEXT_MUTED)
    ax1.text( 1.015, 0, "TE", ha="left",  va="center", fontsize=8, color=TEXT_MUTED)

    ax1.set_xlim(-0.08, 1.22)
    ax1.set_ylim(-0.22, 0.25)
    ax1.set_xlabel("x/c", fontsize=10)
    ax1.set_ylabel("y/c", fontsize=10)
    ax1.set_title("Airfoil cross-section", fontsize=11, pad=8)
    ax1.legend(fontsize=8, loc="upper right", framealpha=0.8)
    ax1.tick_params(labelsize=8)

    rows = list(props.items())
    table = ax1.table(
        cellText=rows,
        colLabels=["Property", "Value"],
        loc="lower left",
        cellLoc="left",
        bbox=[0.0, -0.42, 0.80, 0.32],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(BORDER_COL)
        if r == 0:
            cell.set_facecolor("#21262d")
            cell.set_text_props(fontweight="bold", color=TEXT_COL)
        else:
            cell.set_facecolor(BG_PANEL)
            cell.set_text_props(color=TEXT_MUTED)

    # ── Panel 2: C-mesh domain schematic ────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    _style_ax(ax2)
    ax2.set_aspect("equal")

    R = domain_radius
    cx_domain = 0.5
    theta_ff = np.linspace(0, 2 * np.pi, 300)
    fx = cx_domain + R * np.cos(theta_ff)
    fy = R * np.sin(theta_ff)
    ax2.plot(fx, fy, color=C_DOMAIN, lw=1.5, label=f"Far-field (R={R}c)")
    ax2.fill(fx, fy, alpha=0.04, color=C_DOMAIN)

    wake_rect = mpatches.FancyBboxPatch(
        (0.8, -1.0), 4.2, 2.0,
        boxstyle="square,pad=0", linewidth=1,
        edgecolor=C_WAKE, facecolor=C_WAKE, alpha=0.08, linestyle="--"
    )
    ax2.add_patch(wake_rect)
    ax2.text(2.5, -1.15, f"Wake refinement  Δ={wake_size}c",
             ha="center", fontsize=7.5, color=C_WAKE)

    ax2.fill(
        np.concatenate([xu, xl[::-1]]),
        np.concatenate([yu, yl[::-1]]),
        color=C_AF_FILL, alpha=0.85, zorder=4
    )
    ax2.plot(np.concatenate([xu, xl[::-1], [xu[0]]]),
             np.concatenate([yu, yl[::-1], [yu[0]]]),
             color=C_UPPER, lw=1.2, zorder=5)

    ax2.annotate("", xy=(-R + 1, 0), xytext=(-R + 3.5, 0),
                 arrowprops=dict(arrowstyle="->", color=C_ARROW, lw=1.2))
    ax2.text(-R + 2.2, 0.6, "Inflow\n$U_\\infty$",
             ha="center", fontsize=8, color=C_ARROW)

    le_p = mpatches.Patch(color=C_UPPER,   alpha=0.8, label=f"LE size: {le_size}c")
    wk_p = mpatches.Patch(color=C_WAKE,    alpha=0.8, label=f"Wake size: {wake_size}c")
    ff_p = mpatches.Patch(color=C_DOMAIN,  alpha=0.8, label=f"Far-field size: {far_size}c")
    ax2.legend(handles=[le_p, wk_p, ff_p],
               fontsize=7.5, loc="upper right", framealpha=0.8)

    lim = R * 1.15
    ax2.set_xlim(-lim + cx_domain, lim + cx_domain)
    ax2.set_ylim(-lim, lim)
    ax2.set_xlabel("x/c", fontsize=10)
    ax2.set_ylabel("y/c", fontsize=10)
    ax2.set_title("C-mesh domain layout", fontsize=11, pad=8)
    ax2.tick_params(labelsize=8)

    out_png = f"{slug}_viz.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"[OK] Visualisation saved → {out_png}")
    plt.show()


# ═══════════════════════════════════════════════════════════════════
# Mesh generation
# ═══════════════════════════════════════════════════════════════════

def build_mesh(series_type, key, output_file,
               domain_radius=20.0, le_size=0.005,
               wake_size=0.05, far_size=2.0, n_points=150):
    try:
        import gmsh
    except ImportError:
        print("[ERROR] gmsh not installed.  Run: pip install gmsh")
        sys.exit(1)

    xu, yu, xl, yl, _, _ = get_surfaces(series_type, key, n_points)

    x_surf = np.concatenate([xu[::-1], xl[1:]])
    y_surf = np.concatenate([yu[::-1], yl[1:]])

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    lbl = designation_label(series_type, key).replace(" ", "")
    gmsh.model.add(lbl)

    airfoil_pts = []
    for xi, yi in zip(x_surf, y_surf):
        pt = gmsh.model.geo.addPoint(xi, yi, 0, le_size)
        airfoil_pts.append(pt)

    spline = gmsh.model.geo.addSpline(airfoil_pts + [airfoil_pts[0]])

    cx, cy = 0.5, 0.0
    n_ff = 60
    ff_pts = []
    for i in range(n_ff):
        angle = 2 * np.pi * i / n_ff
        pt = gmsh.model.geo.addPoint(
            cx + domain_radius * np.cos(angle),
            cy + domain_radius * np.sin(angle),
            0, far_size
        )
        ff_pts.append(pt)

    ff_lines = []
    for i in range(n_ff):
        ff_lines.append(
            gmsh.model.geo.addLine(ff_pts[i], ff_pts[(i + 1) % n_ff])
        )

    airfoil_loop  = gmsh.model.geo.addCurveLoop([spline])
    farfield_loop = gmsh.model.geo.addCurveLoop(ff_lines)
    surface       = gmsh.model.geo.addPlaneSurface([farfield_loop, airfoil_loop])

    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(1, [spline],  tag=1, name="airfoil")
    gmsh.model.addPhysicalGroup(1, ff_lines,  tag=2, name="farfield")
    gmsh.model.addPhysicalGroup(2, [surface], tag=3, name="fluid")

    field_box = gmsh.model.mesh.field.add("Box")
    gmsh.model.mesh.field.setNumber(field_box, "VIn",  wake_size)
    gmsh.model.mesh.field.setNumber(field_box, "VOut", far_size)
    gmsh.model.mesh.field.setNumber(field_box, "XMin", 0.8)
    gmsh.model.mesh.field.setNumber(field_box, "XMax", 5.0)
    gmsh.model.mesh.field.setNumber(field_box, "YMin", -1.0)
    gmsh.model.mesh.field.setNumber(field_box, "YMax",  1.0)
    gmsh.model.mesh.field.setAsBackgroundMesh(field_box)

    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.optimize("Netgen")
    gmsh.write(output_file)
    gmsh.finalize()

    print(f"[OK] Mesh written → {output_file}")


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate & visualise a C-mesh around a NACA 4-digit or 6-series airfoil.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "designation", nargs="?", default=None,
        help="NACA code: 4-digit (e.g. 2412) or 6-series (e.g. 66-212)"
    )
    parser.add_argument("--output",    default=None)
    parser.add_argument("--radius",    type=float, default=20.0)
    parser.add_argument("--le-size",   type=float, default=0.005)
    parser.add_argument("--wake-size", type=float, default=0.05)
    parser.add_argument("--far-size",  type=float, default=2.0)
    parser.add_argument("--n-points",  type=int,   default=150)
    parser.add_argument("--no-viz",    action="store_true")
    parser.add_argument("--no-mesh",   action="store_true")
    return parser.parse_args()


def prompt_designation():
    print("Enter NACA designation:")
    print("  4-digit : e.g. 2412")
    print("  6-series: e.g. 66-212  or  64-811")
    while True:
        raw = input("  > ").strip()
        try:
            return parse_designation(raw)
        except ValueError as e:
            print(f"  x {e}")


def main():
    args = parse_args()

    if args.designation is None:
        series_type, key = prompt_designation()
    else:
        try:
            series_type, key = parse_designation(args.designation)
        except ValueError as e:
            print(f"[ERROR] {e}"); sys.exit(1)

    lbl    = designation_label(series_type, key)
    slug   = lbl.replace("NACA ", "NACA").replace(" ", "")
    output = args.output or f"mesh_{slug}.su2"

    print(f"\n  {lbl}")
    for k, v in airfoil_properties(series_type, key).items():
        print(f"    {k:<22} {v}")
    print()

    if not args.no_viz:
        visualise(series_type, key, args.n_points,
                  args.radius, args.le_size, args.wake_size, args.far_size)

    if not args.no_mesh:
        build_mesh(series_type, key, output,
                   args.radius, args.le_size, args.wake_size, args.far_size,
                   args.n_points)


if __name__ == "__main__":
    main()
