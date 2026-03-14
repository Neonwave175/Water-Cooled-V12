"""
generate_mesh_naca.py
----------------------
Generates a C-type unstructured mesh around any NACA 4-digit airfoil,
visualises the profile + mesh topology, and exports a .su2 file.

Requirements:
    pip install gmsh numpy matplotlib

Usage:
    python generate_mesh_naca.py                  # prompts for NACA number
    python generate_mesh_naca.py 2412             # NACA 2412, default mesh params
    python generate_mesh_naca.py 2412 --radius 30 --le-size 0.004
    python generate_mesh_naca.py --help

Arguments:
    designation       4-digit NACA code, e.g. 6407 (default: prompted)
    --output FILE     output .su2 filename  (default: mesh_NACA<XXXX>.su2)
    --radius R        far-field domain radius in chord lengths (default: 20)
    --le-size S       leading-edge mesh size (default: 0.005)
    --wake-size S     wake region mesh size  (default: 0.05)
    --far-size S      far-field mesh size    (default: 2.0)
    --n-points N      airfoil surface points (default: 150)
    --no-viz          skip visualisation
    --no-mesh         skip mesh generation (viz only)
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


# ── NACA 4-digit geometry ─────────────────────────────────────────────────────

def naca4(designation: str, n_points: int = 200):
    """
    Returns upper/lower surface coords + camber line for a NACA 4-digit airfoil.
    """
    if len(designation) != 4 or not designation.isdigit():
        raise ValueError(f"'{designation}' is not a valid 4-digit NACA code.")
    if int(designation[1]) == 0 and int(designation[0]) != 0:
        raise ValueError("Camber position digit must be non-zero for cambered airfoils.")

    m = int(designation[0]) / 100.0   # max camber
    p = int(designation[1]) / 10.0    # camber location (avoid div-by-zero below)
    t = int(designation[2:]) / 100.0  # thickness

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
    xu = x  - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x  + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    return xu, yu, xl, yl, x, yc


def airfoil_properties(designation: str):
    """Return human-readable property dict for a NACA 4-digit designation."""
    m = int(designation[0]) / 100.0
    p = int(designation[1]) / 10.0
    t = int(designation[2:]) / 100.0
    return {
        "Max camber":        f"{m*100:.0f}% chord",
        "Max camber pos.":   f"{p*100:.0f}% chord",
        "Max thickness":     f"{t*100:.0f}% chord",
        "Profile type":      "Symmetric" if m == 0 else "Cambered",
    }


# ── Visualisation ─────────────────────────────────────────────────────────────

def visualise(designation: str, n_points: int,
              domain_radius: float, le_size: float,
              wake_size: float, far_size: float):
    """
    Creates a two-panel figure:
      Left  – airfoil cross-section with annotations
      Right – schematic of the C-mesh domain layout
    """
    xu, yu, xl, yl, xc, yc = naca4(designation, n_points)
    props = airfoil_properties(designation)

    # ── colour palette ──────────────────────────────────────────────────────
    C_UPPER   = "#2B7BB9"
    C_LOWER   = "#C0392B"
    C_CAMBER  = "#888888"
    C_CHORD   = "#CCCCCC"
    C_WAKE    = "#F39C12"
    C_DOMAIN  = "#27AE60"
    C_AIRFOIL = "#2B7BB9"
    BG        = "#FAFAFA"

    fig = plt.figure(figsize=(14, 6), facecolor=BG)
    fig.suptitle(f"NACA {designation}  ·  Airfoil & C-mesh layout",
                 fontsize=14, fontweight="bold", y=0.97, color="#222")

    gs = GridSpec(1, 2, figure=fig, wspace=0.35, left=0.07, right=0.97,
                  top=0.88, bottom=0.10)

    # ── Panel 1: airfoil cross-section ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(BG)
    ax1.set_aspect("equal")

    # Fill upper/lower
    ax1.fill_between(xu, yu, 0, alpha=0.08, color=C_UPPER)
    ax1.fill_between(xl, yl, 0, alpha=0.08, color=C_LOWER)

    # Chord line
    ax1.plot([0, 1], [0, 0], color=C_CHORD, lw=0.8, linestyle="--", zorder=1)

    # Camber line
    ax1.plot(xc, yc, color=C_CAMBER, lw=1.0, linestyle="--",
             label="Camber line", zorder=2)

    # Surfaces
    ax1.plot(xu, yu, color=C_UPPER, lw=2.0, label="Upper surface", zorder=3)
    ax1.plot(xl, yl, color=C_LOWER, lw=2.0, label="Lower surface", zorder=3)

    # TE closing line
    ax1.plot([xu[-1], xl[-1]], [yu[-1], yl[-1]], color="#555", lw=1.2, zorder=3)

    # Max thickness annotation
    thick = yu - np.interp(xu, xl, yl)
    idx_t = np.argmax(thick)
    xt = xu[idx_t]
    yt_top = yu[idx_t]
    yt_bot = np.interp(xt, xl, yl)
    ax1.annotate("", xy=(xt, yt_top), xytext=(xt, yt_bot),
                 arrowprops=dict(arrowstyle="<->", color="#555", lw=0.8))
    ax1.text(xt + 0.015, (yt_top + yt_bot) / 2,
             f"t = {int(designation[2:])}%c",
             fontsize=8, color="#555", va="center")

    # Max camber annotation
    if int(designation[0]) > 0:
        idx_c = np.argmax(yc)
        ax1.annotate("", xy=(xc[idx_c], yc[idx_c]), xytext=(xc[idx_c], 0),
                     arrowprops=dict(arrowstyle="<->", color=C_CAMBER, lw=0.8))
        ax1.text(xc[idx_c] + 0.015, yc[idx_c] / 2,
                 f"m = {designation[0]}%c",
                 fontsize=8, color=C_CAMBER, va="center")

    # LE / TE markers
    ax1.scatter([0, 1], [0, 0], s=30, color="#555", zorder=5)
    ax1.text(-0.015, 0, "LE", ha="right", va="center", fontsize=8, color="#555")
    ax1.text( 1.015, 0, "TE", ha="left",  va="center", fontsize=8, color="#555")

    ax1.set_xlim(-0.08, 1.18)
    ax1.set_ylim(-0.22, 0.25)
    ax1.set_xlabel("x/c", fontsize=10)
    ax1.set_ylabel("y/c", fontsize=10)
    ax1.set_title("Airfoil cross-section", fontsize=11, pad=8)
    ax1.legend(fontsize=8, loc="upper right", framealpha=0.7)
    ax1.tick_params(labelsize=8)

    # Property table
    rows = list(props.items())
    col_labels = ["Property", "Value"]
    table = ax1.table(
        cellText=rows,
        colLabels=col_labels,
        loc="lower left",
        cellLoc="left",
        bbox=[0.0, -0.42, 0.72, 0.32],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        if r == 0:
            cell.set_facecolor("#E8EEF6")
            cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor(BG)

    # ── Panel 2: C-mesh domain schematic ────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(BG)
    ax2.set_aspect("equal")

    R = domain_radius
    cx_domain = 0.5  # centred at mid-chord

    # Far-field circle
    theta_ff = np.linspace(0, 2 * np.pi, 300)
    fx = cx_domain + R * np.cos(theta_ff)
    fy = R * np.sin(theta_ff)
    ax2.plot(fx, fy, color=C_DOMAIN, lw=1.5, label=f"Far-field (R={R}c)")
    ax2.fill(fx, fy, alpha=0.04, color=C_DOMAIN)

    # Wake refinement box
    wake_rect = mpatches.FancyBboxPatch(
        (0.8, -1.0), 4.2, 2.0,
        boxstyle="square,pad=0", linewidth=1,
        edgecolor=C_WAKE, facecolor=C_WAKE, alpha=0.08,
        linestyle="--"
    )
    ax2.add_patch(wake_rect)
    ax2.text(2.5, -1.15, f"Wake refinement  Δ={wake_size}c",
             ha="center", fontsize=7.5, color=C_WAKE)

    # Airfoil (scaled down for context)
    ax2.fill_between(xu, yu, xl,  alpha=0.0)
    ax2.fill(
        np.concatenate([xu, xl[::-1]]),
        np.concatenate([yu, yl[::-1]]),
        color=C_AIRFOIL, alpha=0.7, zorder=4
    )
    ax2.plot(np.concatenate([xu, xl[::-1], [xu[0]]]),
             np.concatenate([yu, yl[::-1], [yu[0]]]),
             color=C_AIRFOIL, lw=1.2, zorder=5)

    # Inflow arrow
    ax2.annotate("", xy=(-R + 1, 0), xytext=(-R + 3.5, 0),
                 arrowprops=dict(arrowstyle="->", color="#888", lw=1.2))
    ax2.text(-R + 2.2, 0.6, "Inflow\n$U_\\infty$",
             ha="center", fontsize=8, color="#888")

    # Mesh size legend boxes
    le_patch  = mpatches.Patch(color="#2B7BB9", alpha=0.6,
                               label=f"LE size: {le_size}c")
    wk_patch  = mpatches.Patch(color=C_WAKE,    alpha=0.6,
                               label=f"Wake size: {wake_size}c")
    ff_patch  = mpatches.Patch(color=C_DOMAIN,  alpha=0.6,
                               label=f"Far-field size: {far_size}c")
    ax2.legend(handles=[le_patch, wk_patch, ff_patch],
               fontsize=7.5, loc="upper right", framealpha=0.7)

    lim = R * 1.15
    ax2.set_xlim(-lim + cx_domain, lim + cx_domain)
    ax2.set_ylim(-lim, lim)
    ax2.set_xlabel("x/c", fontsize=10)
    ax2.set_ylabel("y/c", fontsize=10)
    ax2.set_title("C-mesh domain layout", fontsize=11, pad=8)
    ax2.tick_params(labelsize=8)

    plt.savefig(f"naca{designation}_viz.png", dpi=150, bbox_inches="tight",
                facecolor=BG)
    print(f"[OK] Visualisation saved → naca{designation}_viz.png")
    plt.show()


# ── Mesh generation ───────────────────────────────────────────────────────────

def build_mesh(designation: str,
               output_file: str,
               domain_radius: float = 20.0,
               le_size: float = 0.005,
               wake_size: float = 0.05,
               far_size: float = 2.0,
               n_points: int = 150):
    try:
        import gmsh
    except ImportError:
        print("[ERROR] gmsh not installed. Run: pip install gmsh")
        sys.exit(1)

    xu, yu, xl, yl, _, _ = naca4(designation, n_points=n_points)

    x_surf = np.concatenate([xu[::-1], xl[1:]])
    y_surf = np.concatenate([yu[::-1], yl[1:]])

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add(f"NACA{designation}")

    airfoil_pts = []
    for xi, yi in zip(x_surf, y_surf):
        pt = gmsh.model.geo.addPoint(xi, yi, 0, le_size)
        airfoil_pts.append(pt)

    spline = gmsh.model.geo.addSpline(airfoil_pts + [airfoil_pts[0]])

    cx, cy = 0.5, 0.0
    ff_pts = []
    n_ff = 60
    for i in range(n_ff):
        angle = 2 * np.pi * i / n_ff
        fx = cx + domain_radius * np.cos(angle)
        fy = cy + domain_radius * np.sin(angle)
        pt = gmsh.model.geo.addPoint(fx, fy, 0, far_size)
        ff_pts.append(pt)

    ff_lines = []
    for i in range(n_ff):
        ln = gmsh.model.geo.addLine(ff_pts[i], ff_pts[(i + 1) % n_ff])
        ff_lines.append(ln)

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


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate & visualise a C-mesh around any NACA 4-digit airfoil.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "designation", nargs="?", default=None,
        help="4-digit NACA code, e.g. 6407 or 2412"
    )
    parser.add_argument("--output",     default=None,  help="Output .su2 filename")
    parser.add_argument("--radius",     type=float, default=20.0,
                        metavar="R",   help="Far-field radius in chord lengths (default: 20)")
    parser.add_argument("--le-size",    type=float, default=0.005,
                        metavar="S",   help="Leading-edge cell size (default: 0.005)")
    parser.add_argument("--wake-size",  type=float, default=0.05,
                        metavar="S",   help="Wake cell size (default: 0.05)")
    parser.add_argument("--far-size",   type=float, default=2.0,
                        metavar="S",   help="Far-field cell size (default: 2.0)")
    parser.add_argument("--n-points",   type=int,   default=150,
                        metavar="N",   help="Airfoil surface point count (default: 150)")
    parser.add_argument("--no-viz",     action="store_true",
                        help="Skip visualisation")
    parser.add_argument("--no-mesh",    action="store_true",
                        help="Skip mesh generation (visualise only)")
    return parser.parse_args()


def prompt_designation():
    while True:
        raw = input("Enter NACA 4-digit designation (e.g. 6407): ").strip()
        if len(raw) == 4 and raw.isdigit():
            return raw
        print("  ✗ Must be exactly 4 digits. Try again.")


def main():
    args = parse_args()

    # Get designation interactively if not supplied
    desig = args.designation
    if desig is None:
        desig = prompt_designation()
    else:
        if len(desig) != 4 or not desig.isdigit():
            print(f"[ERROR] '{desig}' is not a valid 4-digit NACA code.")
            sys.exit(1)

    output = args.output or f"mesh_NACA{desig}.su2"

    print(f"\n  NACA {desig}")
    for k, v in airfoil_properties(desig).items():
        print(f"    {k:<22} {v}")
    print()

    if not args.no_viz:
        visualise(
            designation=desig,
            n_points=args.n_points,
            domain_radius=args.radius,
            le_size=args.le_size,
            wake_size=args.wake_size,
            far_size=args.far_size,
        )

    if not args.no_mesh:
        build_mesh(
            designation=desig,
            output_file=output,
            domain_radius=args.radius,
            le_size=args.le_size,
            wake_size=args.wake_size,
            far_size=args.far_size,
            n_points=args.n_points,
        )


if __name__ == "__main__":
    main()
