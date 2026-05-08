"""Parse a dual-cmd play-script log and plot per-step state for V1 and V2.

Usage:
    python scripts/rsl_rl/plot_dual_trace.py /path/to/trace.log [--save out.png]

Expects lines of the form:
    [V1 step=N phi=... t=... total_t=... gate=N traj=N skill=N vel=[x, y, z]]
    [V2 step=N phi=... t=... gate=N traj=N skill=N vel=[x, y, z]]
    [VV step=N v1_V=... v2_V=... dV=+/-... dy_des=... dy_act=...
        ddy_des=... ddy_act=... ref_diff=...]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Per-cmd end-of-step lines.
_LINE_RE = re.compile(
    r"\[(V1|V2) step=(?P<step>\d+) "
    r"phi=(?P<phi>[\d\.eE+-]+) "
    r"t=(?P<t>[\d\.eE+-]+)"
    r"(?: total_t=(?P<total_t>[\d\.eE+-]+))? "
    r"gate=(?P<gate>-?\d+) "
    r"traj=(?P<traj>-?\d+) "
    r"skill=(?P<skill>-?\d+) "
    r"vel=\[(?P<vx>[\d\.eE+-]+), (?P<vy>[\d\.eE+-]+), (?P<vz>[\d\.eE+-]+)\]"
)

# VV comparison line emitted from V1's _update_command override.
# All trailing fields are optional — older logs may omit some.
_VV_RE = re.compile(
    r"\[VV step=(?P<step>\d+) "
    r"v1_V=(?P<v1_V>[\d\.eE+-]+) "
    r"v2_V=(?P<v2_V>[\d\.eE+-]+) "
    r"dV=(?P<dV>[\d\.eE+-]+) "
    r"dy_des=(?P<dy_des>[\d\.eE+-]+) "
    r"dy_act=(?P<dy_act>[\d\.eE+-]+) "
    r"ddy_des=(?P<ddy_des>[\d\.eE+-]+) "
    r"ddy_act=(?P<ddy_act>[\d\.eE+-]+)"
    r"(?: ref_diff=(?P<ref_diff>[\d\.eE+-]+))?"
    r"(?: v1_dom=(?P<v1_dom>-?\d+))?"
    r"(?: v2_dom=(?P<v2_dom>-?\d+))?"
    r"(?: dom_match=(?P<dom_match>True|False))?"
    r"(?: v1_phi=(?P<v1_phi>[\d\.eE+-]+))?"
    r"(?: v2_phi=(?P<v2_phi>[\d\.eE+-]+))?"
    r"(?: v1_fold=(?P<v1_fold>True|False))?"
    r"(?: v2_fold=(?P<v2_fold>True|False))?"
    r"(?: fold_match=(?P<fold_match>True|False))?"
    r"\]"
)


def parse_log(path: Path) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    """Return ({'V1': {...}, 'V2': {...}}, {VV-fields: ...})."""
    fields = ("step", "phi", "t", "gate", "traj", "skill", "vx", "vy", "vz")
    out: dict[str, dict[str, list]] = {
        "V1": {f: [] for f in fields},
        "V2": {f: [] for f in fields},
    }
    vv_fields = ("step", "v1_V", "v2_V", "dV", "dy_des", "dy_act",
                 "ddy_des", "ddy_act", "ref_diff",
                 "v1_dom", "v2_dom", "dom_match",
                 "v1_phi_hp", "v2_phi_hp",
                 "v1_fold", "v2_fold", "fold_match")
    vv: dict[str, list] = {f: [] for f in vv_fields}

    with path.open() as f:
        for line in f:
            m = _LINE_RE.search(line)
            if m is not None:
                tag = m.group(1)
                d = out[tag]
                d["step"].append(int(m.group("step")))
                d["phi"].append(float(m.group("phi")))
                d["t"].append(float(m.group("t")))
                d["gate"].append(int(m.group("gate")))
                d["traj"].append(int(m.group("traj")))
                d["skill"].append(int(m.group("skill")))
                d["vx"].append(float(m.group("vx")))
                d["vy"].append(float(m.group("vy")))
                d["vz"].append(float(m.group("vz")))
                continue

            m = _VV_RE.search(line)
            if m is not None:
                vv["step"].append(int(m.group("step")))
                vv["v1_V"].append(float(m.group("v1_V")))
                vv["v2_V"].append(float(m.group("v2_V")))
                vv["dV"].append(float(m.group("dV")))
                vv["dy_des"].append(float(m.group("dy_des")))
                vv["dy_act"].append(float(m.group("dy_act")))
                vv["ddy_des"].append(float(m.group("ddy_des")))
                vv["ddy_act"].append(float(m.group("ddy_act")))

                def _opt_float(name):
                    g = m.group(name)
                    return float(g) if g is not None else np.nan

                def _opt_int(name):
                    g = m.group(name)
                    return int(g) if g is not None else -999

                def _opt_bool(name):
                    g = m.group(name)
                    if g is None:
                        return np.nan
                    return 1.0 if g == "True" else 0.0

                vv["ref_diff"].append(_opt_float("ref_diff"))
                vv["v1_dom"].append(_opt_int("v1_dom"))
                vv["v2_dom"].append(_opt_int("v2_dom"))
                vv["dom_match"].append(_opt_bool("dom_match"))
                vv["v1_phi_hp"].append(_opt_float("v1_phi"))
                vv["v2_phi_hp"].append(_opt_float("v2_phi"))
                vv["v1_fold"].append(_opt_bool("v1_fold"))
                vv["v2_fold"].append(_opt_bool("v2_fold"))
                vv["fold_match"].append(_opt_bool("fold_match"))

    for tag in out:
        for k in out[tag]:
            out[tag][k] = np.asarray(out[tag][k])
    vv_arr = {k: np.asarray(v) for k, v in vv.items()}
    return out, vv_arr


def aligned_diff(v1_steps, v1_vals, v2_steps, v2_vals):
    """Position-wise diff.  V1 and V2 emit one record per step in lockstep,
    so the i-th record of each corresponds to the same env step.  Avoids
    issues with `step` repeating across episode resets in a multi-episode
    play log."""
    n = min(len(v1_steps), len(v2_steps))
    return np.arange(n), v1_vals[:n] - v2_vals[:n]


def cumulative_steps(steps: np.ndarray) -> np.ndarray:
    """Convert a per-episode step counter (which restarts at 0 on each
    episode reset) into a monotonically increasing global step axis.
    """
    if len(steps) == 0:
        return steps
    cum = steps.astype(np.int64).copy()
    offset = 0
    for i in range(1, len(cum)):
        if steps[i] < steps[i - 1]:
            offset = cum[i - 1] + 1
        cum[i] = steps[i] + offset
    return cum


def reset_indices(steps: np.ndarray) -> np.ndarray:
    """Indices into `steps` where an episode reset occurred (step decreased)."""
    if len(steps) <= 1:
        return np.array([], dtype=np.int64)
    return np.where(np.diff(steps) < 0)[0] + 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", type=Path, help="Path to the trace log file.")
    ap.add_argument("--save", type=Path, default=None,
                    help="Save figure to this path instead of showing.")
    args = ap.parse_args()

    data, vv = parse_log(args.log)
    if data["V1"]["step"].size == 0 and data["V2"]["step"].size == 0 and vv["step"].size == 0:
        print(f"No matching V1/V2/VV lines parsed from {args.log}", file=sys.stderr)
        return 1

    have_vv = vv["step"].size > 0
    have_match = have_vv and not np.all(np.isnan(vv["fold_match"]))
    n_rows = 6 + (3 if have_vv else 0) + (1 if have_match else 0)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 2.2 * n_rows), sharex=True)
    axes_iter = iter(axes)
    ax_phi = next(axes_iter)
    ax_dphi = next(axes_iter)
    ax_traj = next(axes_iter)
    ax_skill = next(axes_iter)
    ax_vel = next(axes_iter)
    ax_gate = next(axes_iter)
    ax_v = next(axes_iter) if have_vv else None
    ax_dout = next(axes_iter) if have_vv else None
    ax_ref = next(axes_iter) if have_vv else None
    ax_match = next(axes_iter) if have_match else None

    # Build a global, monotonic x-axis per data source (per-episode "step"
    # counters reset to 0, so stacking episodes on the raw step axis would
    # cause overlapping/back-tracking lines).
    for tag in ("V1", "V2"):
        d = data[tag]
        d["x"] = cumulative_steps(d["step"]) if d["step"].size else d["step"]
    vv["x"] = cumulative_steps(vv["step"]) if vv["step"].size else vv["step"]

    # Reset markers (vertical lines) — taken from V1 since both cmds reset together.
    reset_xs = (
        data["V1"]["x"][reset_indices(data["V1"]["step"])]
        if data["V1"]["step"].size else np.array([])
    )

    def _mark_resets(ax):
        for x in reset_xs:
            ax.axvline(x, color="k", alpha=0.15, lw=0.7, linestyle="--")

    # --- Per-cmd state plots ----------------------------------------------
    for tag, color in (("V1", "tab:blue"), ("V2", "tab:orange")):
        d = data[tag]
        if d["step"].size == 0:
            continue
        ax_phi.plot(d["x"], d["phi"], label=tag, color=color, lw=1)
        ax_traj.plot(d["x"], d["traj"], label=tag, color=color,
                     drawstyle="steps-post", lw=1)
        ax_skill.plot(d["x"], d["skill"], label=tag, color=color,
                      drawstyle="steps-post", lw=1)
        ax_vel.plot(d["x"], d["vx"], label=f"{tag} vx", color=color, lw=1)
        ax_gate.plot(d["x"], d["gate"], label=tag, color=color,
                     drawstyle="steps-post", lw=1)
    for ax in (ax_phi, ax_traj, ax_skill, ax_vel, ax_gate, ax_dphi):
        _mark_resets(ax)

    if data["V1"]["step"].size and data["V2"]["step"].size:
        _, dphi = aligned_diff(
            data["V1"]["step"], data["V1"]["phi"],
            data["V2"]["step"], data["V2"]["phi"],
        )
        n = len(dphi)
        ax_dphi.plot(data["V1"]["x"][:n], dphi, color="tab:red", lw=1)
        ax_dphi.axhline(0, color="k", lw=0.5, alpha=0.3)
    else:
        ax_dphi.text(0.5, 0.5, "(need both V1 and V2 to diff)",
                     ha="center", va="center", transform=ax_dphi.transAxes)

    ax_phi.set_ylabel("phi"); ax_phi.legend(loc="upper right", fontsize=8); ax_phi.grid(True, alpha=0.3)
    ax_dphi.set_ylabel("V1.phi - V2.phi"); ax_dphi.grid(True, alpha=0.3)
    ax_traj.set_ylabel("traj idx"); ax_traj.legend(loc="upper right", fontsize=8); ax_traj.grid(True, alpha=0.3)
    ax_skill.set_ylabel("skill idx"); ax_skill.legend(loc="upper right", fontsize=8); ax_skill.grid(True, alpha=0.3)
    ax_vel.set_ylabel("commanded vel x"); ax_vel.legend(loc="upper right", fontsize=8); ax_vel.grid(True, alpha=0.3)
    ax_gate.set_ylabel("gate idx"); ax_gate.legend(loc="upper right", fontsize=8); ax_gate.grid(True, alpha=0.3)

    # --- VV comparison plots ----------------------------------------------
    if have_vv:
        x = vv["x"]
        ax_v.plot(x, vv["v1_V"], label="V1.V", color="tab:blue", lw=1)
        ax_v.plot(x, vv["v2_V"], label="V2.V", color="tab:orange", lw=1)
        ax_v.set_ylabel("CLF V")
        ax_v.legend(loc="upper right", fontsize=8)
        ax_v.grid(True, alpha=0.3)
        ax_v_twin = ax_v.twinx()
        ax_v_twin.plot(x, vv["dV"], color="tab:red", lw=0.8, alpha=0.7, label="dV (V1-V2)")
        ax_v_twin.set_ylabel("dV (V1 - V2)", color="tab:red")
        ax_v_twin.tick_params(axis="y", labelcolor="tab:red")

        # Output diffs (positions and velocities) on log scale.
        ax_dout.plot(x, vv["dy_des"],  label="|dy_des|",  color="tab:blue",   lw=1)
        ax_dout.plot(x, vv["dy_act"],  label="|dy_act|",  color="tab:cyan",   lw=1)
        ax_dout.plot(x, vv["ddy_des"], label="|ddy_des|", color="tab:orange", lw=1)
        ax_dout.plot(x, vv["ddy_act"], label="|ddy_act|", color="tab:olive",  lw=1)
        ax_dout.set_yscale("symlog", linthresh=1e-4)
        ax_dout.set_ylabel("output diff norms")
        ax_dout.legend(loc="upper right", fontsize=8, ncol=2)
        ax_dout.grid(True, alpha=0.3, which="both")

        # ref_poses divergence on its own axes (linear, often small).
        if not np.all(np.isnan(vv["ref_diff"])):
            ax_ref.plot(x, vv["ref_diff"], color="tab:purple", lw=1)
            ax_ref.set_ylabel("|V1.ref_poses - V2.ref_poses|")
        else:
            ax_ref.text(0.5, 0.5, "(no ref_diff in log)",
                        ha="center", va="center", transform=ax_ref.transAxes)
            ax_ref.set_ylabel("ref_diff")
        ax_ref.grid(True, alpha=0.3)

        for ax in (ax_v, ax_dout, ax_ref):
            _mark_resets(ax)

    # --- Match indicators (binary 0/1 traces) -----------------------------
    if have_match:
        x = vv["x"]
        # Plot dom_match and fold_match as 0/1 step traces.
        for name, color, offset in (
            ("dom_match", "tab:blue", 0.05),
            ("fold_match", "tab:red", -0.05),
        ):
            arr = vv[name]
            mask = ~np.isnan(arr)
            if mask.any():
                ax_match.plot(x[mask], arr[mask] + offset, label=name,
                              color=color, drawstyle="steps-post", lw=1)
        # Mark mismatch moments with vertical pink lines for fold_match=False.
        fm = vv["fold_match"]
        miss_mask = ~np.isnan(fm) & (fm < 0.5)
        for xm in x[miss_mask]:
            ax_match.axvline(xm, color="tab:red", alpha=0.15, lw=0.7)
        ax_match.set_ylabel("match (1=ok)")
        ax_match.set_ylim(-0.2, 1.2)
        ax_match.legend(loc="upper right", fontsize=8)
        ax_match.grid(True, alpha=0.3)
        _mark_resets(ax_match)

    axes[-1].set_xlabel("global step (cumulative across episodes)")
    fig.suptitle(f"Dual-cmd trace: {args.log.name}", fontsize=11)
    fig.tight_layout()

    if args.save is not None:
        fig.savefig(args.save, dpi=120)
        print(f"Saved {args.save}")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
