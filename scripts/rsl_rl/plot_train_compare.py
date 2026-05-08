"""Plot dual-cmd comparison metrics from one or more training logs.

Usage:
    # one log
    python scripts/rsl_rl/plot_train_compare.py /tmp/v1v2_train_log_1.jsonl

    # compare multiple runs (e.g. V1-primary vs V2-primary)
    python scripts/rsl_rl/plot_train_compare.py \\
        /tmp/v1_primary.jsonl /tmp/v2_primary.jsonl \\
        --labels "V1 primary" "V2 primary" \\
        --save /tmp/compare.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_log(path: Path) -> dict[str, np.ndarray | list]:
    """Parse a JSONL training log.

    Skips ``_meta`` header rows.  Scalar fields become numpy arrays; vector
    fields (lists of floats per row) are returned as a list-of-lists so the
    caller can decide how to summarise them (the plot script ignores them).
    """
    rows: list[dict] = []
    meta: dict | None = None
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("_meta"):
                meta = rec
                continue
            rows.append(rec)
    if not rows:
        return {}
    all_keys: set[str] = set()
    for r in rows:
        all_keys.update(r.keys())
    out: dict[str, np.ndarray | list] = {}
    for k in all_keys:
        sample = next((r[k] for r in rows if k in r), None)
        if isinstance(sample, list):
            # Vector field — keep as list of lists; plot script will skip it.
            out[k] = [r.get(k, []) for r in rows]
        else:
            out[k] = np.asarray([r.get(k, np.nan) for r in rows], dtype=float)
    if meta is not None:
        out["_meta"] = meta
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", type=Path, nargs="+",
                    help="One or more JSONL log files")
    ap.add_argument("--labels", type=str, nargs="*", default=None,
                    help="One label per log (defaults to file stem)")
    ap.add_argument("--save", type=Path, default=None,
                    help="Save figure to this path instead of showing")
    args = ap.parse_args()

    if args.labels is not None and len(args.labels) != len(args.logs):
        print(
            f"Error: --labels has {len(args.labels)} entries but {len(args.logs)} logs were passed",
            file=sys.stderr,
        )
        return 1
    labels = args.labels if args.labels else [p.stem for p in args.logs]

    parsed = [(label, parse_log(path)) for label, path in zip(labels, args.logs)]
    parsed = [(lab, d) for lab, d in parsed if d]  # drop empties
    if not parsed:
        print("No records parsed from any log.", file=sys.stderr)
        return 1

    # Detect optional fields to size the figure.
    have_phi = any("v1_phi" in d for _, d in parsed)
    have_traj = any("v1_traj" in d for _, d in parsed)
    have_skill = any("v1_skill" in d for _, d in parsed)
    have_gate = any("v1_gate_idx" in d for _, d in parsed)
    have_domain = any("v1_domain" in d for _, d in parsed)
    have_fire = any("v1_gate_fired" in d for _, d in parsed)
    have_eplen = any("ep_len" in d for _, d in parsed)
    have_calls = any("shadow_calls_since_log" in d for _, d in parsed)
    n_extra = (
        (2 if have_phi else 0)
        + (1 if have_traj else 0)
        + (1 if have_skill else 0)
        + (1 if have_gate else 0)
        + (1 if have_domain else 0)
        + (1 if have_fire else 0)
        + (1 if have_eplen else 0)
        + (1 if have_calls else 0)
    )
    n_rows = 5 + n_extra
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 2.5 * n_rows), sharex=True)
    axes_iter = iter(axes)
    ax_V = next(axes_iter)
    ax_dV = next(axes_iter)
    ax_dy = next(axes_iter)
    ax_ddy = next(axes_iter)
    ax_ref = next(axes_iter)
    ax_phi = next(axes_iter) if have_phi else None
    ax_dphi = next(axes_iter) if have_phi else None
    ax_traj = next(axes_iter) if have_traj else None
    ax_skill = next(axes_iter) if have_skill else None
    ax_gate = next(axes_iter) if have_gate else None
    ax_domain = next(axes_iter) if have_domain else None
    ax_fire = next(axes_iter) if have_fire else None
    ax_eplen = next(axes_iter) if have_eplen else None
    ax_calls = next(axes_iter) if have_calls else None

    palette = plt.rcParams["axes.prop_cycle"].by_key().get("color",
        ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"])

    for i, (label, data) in enumerate(parsed):
        color = palette[i % len(palette)]
        x = data.get("global_step")
        if x is None:
            continue

        # CLF V: V1 (solid) and shadow (dashed) per log.
        if "v1_V" in data:
            ax_V.plot(x, data["v1_V"], color=color, linestyle="-", lw=1,
                      label=f"{label} V1.V")
        if "shadow_V" in data:
            ax_V.plot(x, data["shadow_V"], color=color, linestyle="--", lw=1,
                      label=f"{label} shadow.V")

        # Signed dV per log.
        if "dV" in data:
            ax_dV.plot(x, data["dV"], color=color, lw=1, label=label)

        # Position output diffs.
        if "dy_des" in data:
            ax_dy.plot(x, data["dy_des"], color=color, linestyle="-", lw=1,
                       label=f"{label} |dy_des|")
        if "dy_act" in data:
            ax_dy.plot(x, data["dy_act"], color=color, linestyle="--", lw=1,
                       label=f"{label} |dy_act|")

        # Velocity output diffs.
        if "ddy_des" in data:
            ax_ddy.plot(x, data["ddy_des"], color=color, linestyle="-", lw=1,
                        label=f"{label} |ddy_des|")
        if "ddy_act" in data:
            ax_ddy.plot(x, data["ddy_act"], color=color, linestyle="--", lw=1,
                        label=f"{label} |ddy_act|")

        # ref_poses divergence.
        if "ref_diff" in data:
            ax_ref.plot(x, data["ref_diff"], color=color, lw=1, label=label)

        # Phase (V1 solid, shadow dashed) and phase diff.
        if ax_phi is not None and "v1_phi" in data:
            ax_phi.plot(x, data["v1_phi"], color=color, linestyle="-", lw=1,
                        label=f"{label} V1.phi")
            if "shadow_phi" in data:
                ax_phi.plot(x, data["shadow_phi"], color=color, linestyle="--",
                            lw=1, label=f"{label} shadow.phi")
        if ax_dphi is not None and "v1_phi" in data and "shadow_phi" in data:
            ax_dphi.plot(x, data["v1_phi"] - data["shadow_phi"],
                         color=color, lw=1, label=label)

        # Trajectory index (step plot).
        if ax_traj is not None and "v1_traj" in data:
            ax_traj.plot(x, data["v1_traj"], color=color, linestyle="-", lw=1,
                         drawstyle="steps-post", label=f"{label} V1.traj")
            if "shadow_traj" in data:
                ax_traj.plot(x, data["shadow_traj"], color=color, linestyle="--",
                             lw=1, drawstyle="steps-post",
                             label=f"{label} shadow.traj")

        # Skill index (step plot).
        if ax_skill is not None and "v1_skill" in data:
            ax_skill.plot(x, data["v1_skill"], color=color, linestyle="-", lw=1,
                          drawstyle="steps-post", label=f"{label} V1.skill")
            if "shadow_skill" in data:
                ax_skill.plot(x, data["shadow_skill"], color=color, linestyle="--",
                              lw=1, drawstyle="steps-post",
                              label=f"{label} shadow.skill")

        # Gate index (step plot).
        if ax_gate is not None and "v1_gate_idx" in data:
            ax_gate.plot(x, data["v1_gate_idx"], color=color, linestyle="-", lw=1,
                         drawstyle="steps-post", label=f"{label} V1.gate_idx")
            if "shadow_gate_idx" in data:
                ax_gate.plot(x, data["shadow_gate_idx"], color=color,
                             linestyle="--", lw=1, drawstyle="steps-post",
                             label=f"{label} shadow.gate_idx")

        # Domain (step plot).
        if ax_domain is not None and "v1_domain" in data:
            ax_domain.plot(x, data["v1_domain"], color=color, linestyle="-", lw=1,
                           drawstyle="steps-post", label=f"{label} V1.domain")
            if "shadow_domain" in data:
                ax_domain.plot(x, data["shadow_domain"], color=color,
                               linestyle="--", lw=1, drawstyle="steps-post",
                               label=f"{label} shadow.domain")

        # Gate fires + domain changes (markers, offset on y so V1 and
        # shadow events don't overlap).  Lifts the misalignment to the
        # foreground: a single tick gap between v1's fire and shadow's
        # fire shows up as adjacent markers at +1 and -1.
        if ax_fire is not None and "v1_gate_fired" in data:
            v1_fire_x = x[np.asarray(data["v1_gate_fired"]) > 0]
            shadow_fire_x = (
                x[np.asarray(data["shadow_gate_fired"]) > 0]
                if "shadow_gate_fired" in data else np.array([])
            )
            v1_dom_x = (
                x[np.asarray(data["v1_domain_changed"]) > 0]
                if "v1_domain_changed" in data else np.array([])
            )
            shadow_dom_x = (
                x[np.asarray(data["shadow_domain_changed"]) > 0]
                if "shadow_domain_changed" in data else np.array([])
            )
            ax_fire.scatter(v1_fire_x, np.full_like(v1_fire_x, +1.0),
                            color=color, marker="|", s=80,
                            label=f"{label} V1 gate-fire")
            ax_fire.scatter(shadow_fire_x, np.full_like(shadow_fire_x, -1.0),
                            color=color, marker="|", s=80, alpha=0.6,
                            label=f"{label} shadow gate-fire")
            ax_fire.scatter(v1_dom_x, np.full_like(v1_dom_x, +0.5),
                            color=color, marker="x", s=40,
                            label=f"{label} V1 dom-change")
            ax_fire.scatter(shadow_dom_x, np.full_like(shadow_dom_x, -0.5),
                            color=color, marker="x", s=40, alpha=0.6,
                            label=f"{label} shadow dom-change")

        # ep_len trace (shows env-tick advancement vs logger-call cadence).
        if ax_eplen is not None and "ep_len" in data:
            ax_eplen.plot(x, data["ep_len"], color=color, linestyle="-", lw=1,
                          drawstyle="steps-post", label=f"{label} ep_len")

        # Per-record V2 _compute_time call accounting.
        if ax_calls is not None and "shadow_calls_since_log" in data:
            ax_calls.plot(x, data["shadow_calls_since_log"], color=color,
                          linestyle="-", lw=1, label=f"{label} V2 calls")
            if "shadow_idem_hits_since_log" in data:
                ax_calls.plot(x, data["shadow_idem_hits_since_log"], color=color,
                              linestyle="--", lw=1, label=f"{label} V2 idem hits")
            if "shadow_advance_count_since_log" in data:
                ax_calls.plot(x, data["shadow_advance_count_since_log"], color=color,
                              linestyle=":", lw=1.2,
                              label=f"{label} V2 advances")

    # Cosmetics.
    ax_V.set_ylabel("CLF V")
    ax_V.legend(loc="upper right", fontsize=8, ncol=max(1, len(parsed)))
    ax_V.grid(True, alpha=0.3)

    ax_dV.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax_dV.set_ylabel("V1.V − shadow.V")
    ax_dV.legend(loc="upper right", fontsize=8)
    ax_dV.grid(True, alpha=0.3)

    ax_dy.set_yscale("symlog", linthresh=1e-4)
    ax_dy.set_ylabel("position diff norms")
    ax_dy.legend(loc="upper right", fontsize=8, ncol=max(1, len(parsed)))
    ax_dy.grid(True, alpha=0.3, which="both")

    ax_ddy.set_yscale("symlog", linthresh=1e-4)
    ax_ddy.set_ylabel("velocity diff norms")
    ax_ddy.legend(loc="upper right", fontsize=8, ncol=max(1, len(parsed)))
    ax_ddy.grid(True, alpha=0.3, which="both")

    ax_ref.set_ylabel("|V1.ref_poses − shadow.ref_poses|")
    ax_ref.legend(loc="upper right", fontsize=8)
    ax_ref.grid(True, alpha=0.3)

    if ax_phi is not None:
        ax_phi.set_ylabel("phi")
        ax_phi.legend(loc="upper right", fontsize=8, ncol=max(1, len(parsed)))
        ax_phi.grid(True, alpha=0.3)

    if ax_dphi is not None:
        ax_dphi.axhline(0, color="k", lw=0.5, alpha=0.3)
        ax_dphi.set_ylabel("V1.phi − shadow.phi")
        ax_dphi.legend(loc="upper right", fontsize=8)
        ax_dphi.grid(True, alpha=0.3)

    if ax_traj is not None:
        ax_traj.set_ylabel("traj idx")
        ax_traj.legend(loc="upper right", fontsize=8, ncol=max(1, len(parsed)))
        ax_traj.grid(True, alpha=0.3)

    if ax_skill is not None:
        ax_skill.set_ylabel("skill idx")
        ax_skill.legend(loc="upper right", fontsize=8, ncol=max(1, len(parsed)))
        ax_skill.grid(True, alpha=0.3)

    if ax_gate is not None:
        ax_gate.set_ylabel("gate idx")
        ax_gate.legend(loc="upper right", fontsize=8, ncol=max(1, len(parsed)))
        ax_gate.grid(True, alpha=0.3)

    if ax_domain is not None:
        ax_domain.set_ylabel("domain")
        ax_domain.legend(loc="upper right", fontsize=8, ncol=max(1, len(parsed)))
        ax_domain.grid(True, alpha=0.3)

    if ax_fire is not None:
        ax_fire.axhline(0, color="k", lw=0.5, alpha=0.3)
        ax_fire.set_ylim(-1.5, 1.5)
        ax_fire.set_yticks([-1, -0.5, 0.5, 1.0])
        ax_fire.set_yticklabels(
            ["shadow gate-fire", "shadow dom-Δ", "V1 dom-Δ", "V1 gate-fire"]
        )
        ax_fire.set_ylabel("fire / dom Δ")
        ax_fire.grid(True, alpha=0.3, axis="x")

    if ax_eplen is not None:
        ax_eplen.set_ylabel("ep_len (env 0)")
        ax_eplen.legend(loc="upper right", fontsize=8, ncol=max(1, len(parsed)))
        ax_eplen.grid(True, alpha=0.3)

    if ax_calls is not None:
        ax_calls.set_ylabel("V2 _compute_time calls\n(since last log)")
        ax_calls.legend(loc="upper right", fontsize=8, ncol=max(1, len(parsed)))
        ax_calls.grid(True, alpha=0.3)

    axes[-1].set_xlabel("global compute step (env 0)")

    if len(parsed) == 1:
        fig.suptitle(f"Train compare: {args.logs[0].name}", fontsize=11)
    else:
        fig.suptitle(f"Train compare ({len(parsed)} runs)", fontsize=11)
    fig.tight_layout()

    if args.save is not None:
        fig.savefig(args.save, dpi=120)
        print(f"Saved {args.save}")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
