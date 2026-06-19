# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy", "matplotlib", "pillow"]
# ///
"""
Molecular Dynamics (MD) of a Lennard-Jones fluid — from scratch.
=================================================================

This is a complete, dependency-light atom simulator you can read top to bottom.
It simulates N point atoms interacting through the Lennard-Jones (LJ) pair
potential and integrates Newton's equations of motion with the velocity-Verlet
algorithm. Everything is in 2D by default so it animates nicely, but the code is
written dimension-agnostically — set DIM = 3 for a 3D simulation.

The physics, in five pieces
---------------------------
1. POTENTIAL — the Lennard-Jones potential models neutral atoms (e.g. argon):

       U(r) = 4 eps [ (sigma/r)^12  -  (sigma/r)^6 ]

   The r^-12 term is steep short-range repulsion (Pauli exclusion); the r^-6
   term is the attractive van der Waals tail. U has a minimum of depth -eps at
   r = 2^(1/6) sigma.

2. FORCE — force is minus the gradient of the energy. For a pair i, j separated
   by the vector r_ij = r_i - r_j (|r_ij| = r):

       F_ij = 24 eps [ 2 (sigma/r)^12 - (sigma/r)^6 ] * r_ij / r^2

   and F_ij acts on i (with -F_ij on j by Newton's third law).

3. INTEGRATOR — velocity-Verlet, the workhorse of MD. It is time-reversible and
   symplectic, so total energy stays bounded (no slow drift) for small dt:

       v(t+dt/2) = v(t) + (dt/2) a(t)
       x(t+dt)   = x(t) + dt v(t+dt/2)
       a(t+dt)   = F(x(t+dt)) / m
       v(t+dt)   = v(t+dt/2) + (dt/2) a(t+dt)

4. PERIODIC BOUNDARY CONDITIONS (PBC) — a small box mimics bulk material: an
   atom leaving one side re-enters the opposite side, and each pair interacts
   through the *minimum image* (nearest periodic copy). Valid when box_L > 2*rcut.

5. THERMOSTAT — to simulate a fixed temperature (the NVT ensemble) we gently
   rescale velocities toward a target T. With the thermostat OFF you get NVE
   (constant energy) — the cleanest test that the integrator is correct.

Reduced (Lennard-Jones) units
-----------------------------
We set eps = sigma = mass = k_B = 1. Then every quantity is dimensionless:
length in sigma, energy in eps, time in sigma*sqrt(m/eps), temperature in
eps/k_B. For argon: sigma ~ 0.34 nm, eps/k_B ~ 120 K, time unit ~ 2.2 ps.

Run it
------
    uv run lj_md.py                         # defaults: 100 atoms, NVT @ T=1.0
    uv run lj_md.py --thermostat none       # NVE — watch total energy stay flat
    uv run lj_md.py --animate               # live window instead of saving a GIF
    uv run lj_md.py --n-side 16 --temp 2.0  # bigger, hotter (more gas-like)

Outputs (saved next to this script): energy_plot.png and trajectory.gif
"""
from __future__ import annotations

import argparse

import numpy as np

# Spatial dimensionality. 2 is easy to visualize; set to 3 for a 3D fluid.
DIM = 2


# --------------------------------------------------------------------------- #
#  Core physics: forces and energies                                          #
# --------------------------------------------------------------------------- #
def lj_forces_and_pe(positions: np.ndarray, box_l: float, rcut: float):
    """Compute LJ forces and potential energy under minimum-image PBC.

    Parameters
    ----------
    positions : (N, DIM) array of atom coordinates inside the box.
    box_l     : side length of the (cubic/square) periodic box.
    rcut      : interaction cutoff radius (pairs farther apart are ignored).

    Returns
    -------
    forces : (N, DIM) array — net force on each atom.
    pe     : float        — total potential energy (cutoff-shifted).

    This is the O(N^2) all-pairs version: simple, vectorized, and perfectly
    fine for a few hundred atoms. Production codes use neighbor lists / cells.
    """
    # Pairwise displacement vectors r_ij = r_i - r_j, shape (N, N, DIM).
    disp = positions[:, None, :] - positions[None, :, :]
    # Minimum-image convention: fold each component into [-L/2, L/2).
    disp -= box_l * np.round(disp / box_l)

    # Squared distances, shape (N, N). Put +inf on the diagonal so an atom
    # never interacts with itself (and 1/r^2 there is harmlessly zero).
    r2 = np.sum(disp * disp, axis=-1)
    np.fill_diagonal(r2, np.inf)

    # Apply the cutoff: only pairs with r < rcut contribute.
    within = r2 < rcut * rcut
    inv_r2 = np.where(within, 1.0 / r2, 0.0)
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2

    # Force prefactor f_ij such that F_ij = f_ij * r_ij  (see module docstring).
    #   f_ij = 24 (2 r^-12 - r^-6) / r^2
    f_coeff = 24.0 * (2.0 * inv_r12 - inv_r6) * inv_r2  # (N, N)
    # Sum contributions from all partners j onto each atom i.
    forces = np.einsum("ij,ijd->id", f_coeff, disp)  # (N, DIM)

    # Potential energy, shifted so U(rcut) = 0 (removes a jump at the cutoff).
    u_shift = 4.0 * ((1.0 / rcut ** 12) - (1.0 / rcut ** 6))
    u_pair = np.where(within, 4.0 * (inv_r12 - inv_r6) - u_shift, 0.0)
    pe = 0.5 * np.sum(u_pair)  # 0.5 because each pair is counted twice

    return forces, pe


def kinetic_energy(velocities: np.ndarray) -> float:
    """KE = 1/2 sum m v^2, with m = 1 in reduced units."""
    return 0.5 * float(np.sum(velocities * velocities))


def temperature(velocities: np.ndarray) -> float:
    """Instantaneous temperature from equipartition: KE = (Nf/2) k_B T.

    Nf = DIM*N - DIM removes the DIM center-of-mass degrees of freedom that we
    zero out at initialization (they carry no thermal energy).
    """
    n = len(velocities)
    n_dof = DIM * n - DIM
    return 2.0 * kinetic_energy(velocities) / n_dof


# --------------------------------------------------------------------------- #
#  Initialization                                                             #
# --------------------------------------------------------------------------- #
def init_positions(n_side: int, density: float):
    """Place n_side**DIM atoms on a regular lattice at the given number density.

    Starting on a lattice (rather than at random) avoids atom overlaps that
    would create enormous repulsive forces and blow up the integrator.
    """
    spacing = density ** (-1.0 / DIM)  # rho = 1/spacing^DIM  ->  spacing
    box_l = n_side * spacing
    grid = np.arange(n_side) * spacing
    # Cartesian product of the per-axis grids -> (n_side**DIM, DIM).
    mesh = np.meshgrid(*([grid] * DIM), indexing="ij")
    positions = np.stack([m.ravel() for m in mesh], axis=-1).astype(float)
    # Nudge off the perfect lattice so it melts instead of sitting frozen.
    positions += 0.01 * spacing
    return positions, box_l


def init_velocities(n: int, target_t: float, rng: np.random.Generator):
    """Draw Maxwell-Boltzmann velocities, remove drift, and set the temperature.

    Maxwell-Boltzmann in each Cartesian component is just a Gaussian. We then
    (a) subtract the mean so the whole system has zero net momentum, and
    (b) rescale so the instantaneous temperature equals target_t exactly.
    """
    velocities = rng.standard_normal((n, DIM))
    velocities -= velocities.mean(axis=0)  # zero center-of-mass momentum
    if target_t > 0:
        velocities *= np.sqrt(target_t / temperature(velocities))
    return velocities


# --------------------------------------------------------------------------- #
#  Time integration                                                           #
# --------------------------------------------------------------------------- #
def run_simulation(
    n_side: int = 10,
    density: float = 0.8,
    target_t: float = 1.0,
    dt: float = 0.005,
    steps: int = 2000,
    rcut: float = 2.5,
    thermostat: str = "rescale",
    thermo_every: int = 20,
    sample_every: int = 10,
    seed: int = 0,
):
    """Integrate the system and return trajectory frames + energy time series."""
    rng = np.random.default_rng(seed)
    positions, box_l = init_positions(n_side, density)
    n = len(positions)
    if box_l <= 2.0 * rcut:
        raise ValueError(
            f"Box (L={box_l:.2f}) too small for cutoff rcut={rcut}; "
            f"need L > 2*rcut. Use more atoms (--n-side) or lower --density."
        )
    velocities = init_velocities(n, target_t, rng)
    forces, pe = lj_forces_and_pe(positions, box_l, rcut)

    # History buffers.
    frames_xy, frames_speed = [], []
    t_hist, ke_hist, pe_hist = [], [], []

    for step in range(steps):
        # --- velocity-Verlet ---
        velocities += 0.5 * dt * forces            # half kick
        positions += dt * velocities               # drift
        positions %= box_l                          # wrap back into the box (PBC)
        forces, pe = lj_forces_and_pe(positions, box_l, rcut)  # new forces
        velocities += 0.5 * dt * forces            # second half kick

        # --- optional thermostat (velocity rescaling toward target_t) ---
        if thermostat == "rescale" and target_t > 0 and step % thermo_every == 0:
            velocities *= np.sqrt(target_t / temperature(velocities))

        # --- record diagnostics & animation frames ---
        if step % sample_every == 0:
            ke = kinetic_energy(velocities)
            t_hist.append(temperature(velocities))
            ke_hist.append(ke)
            pe_hist.append(pe)
            frames_xy.append(positions.copy())
            frames_speed.append(np.linalg.norm(velocities, axis=1))

    return {
        "box_l": box_l,
        "n": n,
        "frames_xy": frames_xy,
        "frames_speed": frames_speed,
        "temperature": np.asarray(t_hist),
        "kinetic": np.asarray(ke_hist),
        "potential": np.asarray(pe_hist),
        "sample_every": sample_every,
        "dt": dt,
    }


# --------------------------------------------------------------------------- #
#  Reporting & visualization                                                  #
# --------------------------------------------------------------------------- #
def report(result: dict, thermostat: str) -> None:
    """Print a summary, including the energy-conservation drift metric."""
    ke = result["kinetic"]
    pe = result["potential"]
    total = ke + pe
    temp = result["temperature"]

    # Relative energy drift over the second half (after any transient).
    half = len(total) // 2
    tail = total[half:]
    drift = float(np.std(tail) / (abs(np.mean(tail)) + 1e-12))

    print(f"  atoms (N)           : {result['n']}")
    print(f"  box length (L)      : {result['box_l']:.3f}")
    print(f"  samples recorded    : {len(total)}")
    print(f"  mean temperature    : {temp.mean():.4f}  (target set via --temp)")
    print(f"  mean total energy   : {total.mean():.4f}")
    print(f"  energy std (2nd half): {tail.std():.5f}")
    print(f"  relative drift      : {drift:.2e}", end="")
    if thermostat == "none":
        verdict = "excellent" if drift < 1e-2 else ("ok" if drift < 5e-2 else "high")
        print(f"   <- NVE energy conservation: {verdict}")
    else:
        print("   (thermostat ON, so energy is not conserved by design)")


def save_energy_plot(result: dict, path: str) -> None:
    """Save KE / PE / total-energy and temperature vs. time as a PNG."""
    import matplotlib
    matplotlib.use("Agg")  # headless-safe backend
    import matplotlib.pyplot as plt

    ke, pe = result["kinetic"], result["potential"]
    total = ke + pe
    t_axis = np.arange(len(total)) * result["sample_every"] * result["dt"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(t_axis, ke, label="kinetic", color="tab:red", lw=1.2)
    ax1.plot(t_axis, pe, label="potential", color="tab:blue", lw=1.2)
    ax1.plot(t_axis, total, label="total", color="black", lw=1.6)
    ax1.set_xlabel("time (reduced units)")
    ax1.set_ylabel("energy")
    ax1.set_title("Energy vs. time")
    ax1.legend(frameon=False)

    ax2.plot(t_axis, result["temperature"], color="tab:green", lw=1.2)
    ax2.set_xlabel("time (reduced units)")
    ax2.set_ylabel("temperature")
    ax2.set_title("Temperature vs. time")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved energy plot   : {path}")


def visualize_trajectory(result: dict, animate: bool, gif_path: str) -> None:
    """Animate atoms as a scatter plot, colored by speed. Save a GIF or show live."""
    import matplotlib
    if not animate:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    box_l = result["box_l"]
    frames_xy = result["frames_xy"]
    frames_speed = result["frames_speed"]
    vmax = max(float(s.max()) for s in frames_speed) or 1.0

    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.set_xlim(0, box_l)
    ax.set_ylim(0, box_l)
    ax.set_aspect("equal")
    ax.set_title("Lennard-Jones fluid (color = speed)")
    scat = ax.scatter(
        frames_xy[0][:, 0], frames_xy[0][:, 1],
        c=frames_speed[0], cmap="plasma", vmin=0, vmax=vmax, s=80, edgecolors="k",
    )

    def update(i):
        xy = frames_xy[i]
        scat.set_offsets(xy[:, :2])
        scat.set_array(frames_speed[i])
        ax.set_xlabel(f"frame {i + 1}/{len(frames_xy)}")
        return (scat,)

    anim = FuncAnimation(fig, update, frames=len(frames_xy), interval=40, blit=False)

    if animate:
        plt.show()
    else:
        anim.save(gif_path, writer=PillowWriter(fps=25))
        plt.close(fig)
        print(f"  saved animation     : {gif_path}")


# --------------------------------------------------------------------------- #
#  Command-line interface                                                     #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="From-scratch Lennard-Jones molecular dynamics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-side", type=int, default=10,
                   help=f"atoms per axis (total N = n_side**{DIM})")
    p.add_argument("--density", type=float, default=0.8, help="number density (reduced)")
    p.add_argument("--temp", type=float, default=1.0, help="target temperature (reduced)")
    p.add_argument("--dt", type=float, default=0.005, help="time step (reduced)")
    p.add_argument("--steps", type=int, default=2000, help="number of MD steps")
    p.add_argument("--rcut", type=float, default=2.5, help="LJ interaction cutoff")
    p.add_argument("--thermostat", choices=["rescale", "none"], default="rescale",
                   help="'rescale' = NVT (hold T); 'none' = NVE (conserve energy)")
    p.add_argument("--seed", type=int, default=0, help="RNG seed")
    p.add_argument("--animate", action="store_true",
                   help="open a live window instead of saving a GIF")
    return p.parse_args()


def main() -> None:
    import os
    args = parse_args()
    here = os.path.dirname(os.path.abspath(__file__))

    print("Lennard-Jones molecular dynamics")
    print(f"  thermostat = {args.thermostat}, target T = {args.temp}, "
          f"dt = {args.dt}, steps = {args.steps}")
    print("  integrating...")
    result = run_simulation(
        n_side=args.n_side, density=args.density, target_t=args.temp, dt=args.dt,
        steps=args.steps, rcut=args.rcut, thermostat=args.thermostat, seed=args.seed,
    )
    report(result, args.thermostat)
    save_energy_plot(result, os.path.join(here, "energy_plot.png"))
    visualize_trajectory(result, args.animate, os.path.join(here, "trajectory.gif"))
    print("done.")


if __name__ == "__main__":
    main()
