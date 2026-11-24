from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import improvement_percent, metrics
from model import MechanicalParameters, simulate
from policies import StaticImpedancePolicy, build_adaptive_policy
from reference import GaitReference, build_subject_stride_references


SEED = 20260916
DATA = ROOT / "data"
CSV = ROOT / "results" / "csv"
PLOTS = ROOT / "results" / "plots"
VIDEOS = ROOT / "results" / "videos"
DOCS = ROOT / "docs"


def ensure_dirs():
    for p in (CSV, PLOTS, VIDEOS, DOCS, ROOT / "config"):
        p.mkdir(parents=True, exist_ok=True)


def geometry_from_data():
    s = pd.read_csv(DATA / "dogG_subject_summary.csv")
    return float(s["shank_length_m_fixed"].median()), float(s["paw_hock_to_toe_length_m_fixed"].median())


def load_mean_refs(mech):
    shank, paw = geometry_from_data()
    return {
        gait: GaitReference.from_standard_csv(
            DATA / "dogG_personalized_reference_101.csv", gait, shank, paw, harmonics=6
        )
        for gait in ("walk", "trot")
    }


def natural_frame(sim_df):
    d = sim_df.copy()
    d["condition"] = "natural_reference"
    d["q_paw_rad_sim"] = d["q_paw_rad_ref"]
    d["hock_angle_rad_sim"] = d["hock_angle_rad_ref"]
    for sim_col, ref_col in [
        ("toe_x_m_sim", "toe_x_m_ref"), ("toe_y_m_sim", "toe_y_m_ref"),
        ("toe_vx_m_s_sim", "toe_vx_m_s_ref"), ("toe_vy_m_s_sim", "toe_vy_m_s_ref"),
        ("toe_ax_m_s2_sim", "toe_ax_m_s2_ref"), ("toe_ay_m_s2_sim", "toe_ay_m_s2_ref"),
    ]:
        d[sim_col] = d[ref_col]
    dynamic_cols = [
        c for c in d.columns if c.startswith(("k_", "b_", "theta_eq", "tau_"))
        or c in {"normal_force_N", "paw_compression_m", "equilibrium_tuning_power_W", "joint_power_W"}
    ]
    d[dynamic_cols] = np.nan
    return d


def optimize_policies(refs, mech, quick=False):
    trace = []
    eval_counter = {"n": 0}
    steps = 54 if quick else 84
    samples = 49 if quick else 69
    local_passes = 1 if quick else 2

    def record(search, gait, x, value):
        eval_counter["n"] += 1
        trace.append({
            "evaluation": eval_counter["n"], "search": search, "gait_type": gait,
            "objective": float(value), "parameters_json": json.dumps([float(v) for v in x]),
        })

    def static_policy(x):
        return StaticImpedancePolicy(float(x[0]), float(x[1]), float(x[2]), float(x[3]))

    def bounded_search(objective, bounds, n_random, seed, initial=None):
        """Reproducible global space-filling search plus coordinate refinement."""
        lo = np.array([b[0] for b in bounds], float)
        hi = np.array([b[1] for b in bounds], float)
        sampler = qmc.LatinHypercube(d=len(bounds), seed=seed)
        candidates = qmc.scale(sampler.random(n_random), lo, hi)
        if initial is not None:
            candidates = np.vstack([np.asarray(initial, float), candidates])
        values = np.array([objective(x) for x in candidates], float)
        best = candidates[int(np.argmin(values))].copy()
        best_value = float(values.min())
        for pass_no in range(local_passes):
            step = (0.10 if pass_no == 0 else 0.04) * (hi - lo)
            for j in range(len(bounds)):
                for sign in (-1.0, 1.0):
                    x = best.copy(); x[j] = np.clip(x[j] + sign * step[j], lo[j], hi[j])
                    value = float(objective(x))
                    if value < best_value:
                        best, best_value = x, value
        return best, best_value

    static_bounds = [
        (15.0, 137.5098708314), (0.05, 1.35), (np.deg2rad(-160.0), np.deg2rad(-95.0)), (8000.0, 50000.0)
    ]

    def common_obj(x):
        p = static_policy(x)
        vals = []
        for gait in ("walk", "trot"):
            d = simulate(refs[gait], p, mech, steps, samples, settling_cycles=2)
            vals.append(metrics(d, refs[gait], mech)["objective"])
        value = float(np.mean(vals))
        record("common_static", "both", x, value)
        return value

    common_x, _ = bounded_search(
        common_obj, static_bounds, 12 if quick else 42, SEED,
        initial=[105.0, 0.55, np.deg2rad(-128.0), 18000.0],
    )
    common_static = static_policy(common_x)

    gait_static = {}
    for gi, gait in enumerate(("walk", "trot")):
        def objective(x, gait=gait):
            p = static_policy(x)
            d = simulate(refs[gait], p, mech, steps, samples, settling_cycles=2)
            value = metrics(d, refs[gait], mech)["objective"]
            record("gait_specific_static", gait, x, value)
            return value
        x, _ = bounded_search(
            objective, static_bounds, 10 if quick else 32, SEED + 10 + gi,
            initial=[common_static.k_hock, common_static.b_hock, common_static.theta_eq, common_static.k_paw],
        )
        gait_static[gait] = static_policy(x)

    adaptive_bounds = [
        (35.0, 137.5098708314),  # k stance
        (25.0, 137.5098708314),  # k swing
        (0.30, 1.10),            # damping ratio stance
        (0.25, 1.00),            # damping ratio swing
        (0.55, 1.05),            # natural ROM scale
        (0.35, 1.00),            # inverse-dynamics feed-forward gain
        (0.006, 0.018),          # target paw compression
        (0.000, 0.070),          # phase lead fraction
    ]

    def to_adaptive_dict(x):
        return {
            "k_stance": float(x[0]), "k_swing": float(x[1]),
            "zeta_stance": float(x[2]), "zeta_swing": float(x[3]),
            "theta_scale": float(x[4]), "feedforward_gain": float(x[5]),
            "compression_target_m": float(x[6]), "lead_fraction": float(x[7]),
            "k_paw_preload": 15000.0,
        }

    adaptive_pars, adaptive_policy = {}, {}
    for gi, gait in enumerate(("walk", "trot")):
        def objective(x, gait=gait):
            pars = to_adaptive_dict(x)
            p = build_adaptive_policy(refs[gait], mech, pars)
            d = simulate(refs[gait], p, mech, steps, samples, settling_cycles=2)
            m = metrics(d, refs[gait], mech)
            # Small regularization discourages unnecessarily aggressive tracking hardware.
            reg = 0.002 * max(0.0, pars["theta_scale"] - 1.0) ** 2
            value = m["objective"] + reg
            record("adaptive_impedance", gait, x, value)
            return value
        initial = [105.0, 95.0, 0.60, 0.55, 0.85 if gait == "trot" else 0.95, 0.85, 0.010, 0.015]
        x, _ = bounded_search(
            objective, adaptive_bounds, 14 if quick else 52, SEED + 20 + gi, initial=initial,
        )
        adaptive_pars[gait] = to_adaptive_dict(x)
        adaptive_policy[gait] = build_adaptive_policy(refs[gait], mech, adaptive_pars[gait])

    pd.DataFrame(trace).to_csv(CSV / "optimization_trace.csv", index=False)
    return common_static, gait_static, adaptive_pars, adaptive_policy


def evaluate_final(refs, mech, common_static, gait_static, adaptive_policy):
    sims = {}
    rows = []
    for gait in ("walk", "trot"):
        for condition, policy in [
            ("static_single_device", common_static),
            ("static_gait_specific", gait_static[gait]),
            ("adaptive_impedance", adaptive_policy[gait]),
        ]:
            d = simulate(refs[gait], policy, mech, steps_per_cycle=720, samples_per_cycle=401, settling_cycles=5)
            sims[(gait, condition)] = d
            rows.append({"gait_type": gait, "condition": condition, **metrics(d, refs[gait], mech)})
        combined = []
        base = sims[(gait, "static_single_device")]
        combined.append(natural_frame(base))
        for condition in ("static_single_device", "adaptive_impedance"):
            x = sims[(gait, condition)].copy()
            x["condition"] = condition
            combined.append(x)
        pd.concat(combined, ignore_index=True).to_csv(CSV / f"trajectory_comparison_{gait}.csv", index=False)
    summary = pd.DataFrame(rows)
    improvements = []
    for gait in ("walk", "trot"):
        st = summary[(summary.gait_type == gait) & (summary.condition == "static_single_device")].iloc[0]
        gs = summary[(summary.gait_type == gait) & (summary.condition == "static_gait_specific")].iloc[0]
        ad = summary[(summary.gait_type == gait) & (summary.condition == "adaptive_impedance")].iloc[0]
        improvements.append({
            "gait_type": gait,
            "objective_improvement_vs_single_static_percent": improvement_percent(st.objective, ad.objective),
            "objective_improvement_vs_gait_specific_static_percent": improvement_percent(gs.objective, ad.objective),
            "angle_rmse_reduction_vs_single_static_percent": improvement_percent(st.hock_angle_rmse_deg, ad.hock_angle_rmse_deg),
            "toe_rmse_reduction_vs_single_static_percent": improvement_percent(st.toe_position_rmse_mm, ad.toe_position_rmse_mm),
            "jerk_error_reduction_vs_single_static_percent": improvement_percent(st.jerk_tracking_nrmse, ad.jerk_tracking_nrmse),
            "peak_load_error_reduction_vs_single_static_percent": improvement_percent(st.peak_load_relative_error, ad.peak_load_relative_error),
        })
    imp = pd.DataFrame(improvements)
    summary.to_csv(CSV / "summary_metrics.csv", index=False)
    imp.to_csv(CSV / "improvement_summary.csv", index=False)

    profiles = []
    for gait in ("walk", "trot"):
        d = sims[(gait, "adaptive_impedance")]
        profiles.append(d[[
            "cycle_percent", "phase", "k_hock_command_Nm_rad", "b_hock_command_Nm_s_rad",
            "theta_eq_command_rad", "k_paw_command_N_m",
        ]].assign(gait_type=gait))
    pd.concat(profiles, ignore_index=True).to_csv(CSV / "adaptive_controller_profiles.csv", index=False)
    return sims, summary, imp


def heldout_validation(mech, refs, common_static, adaptive_policy):
    shank, paw = geometry_from_data()
    strides = build_subject_stride_references(
        DATA / "Canine_Prosthesis_Kinematic_Targets_101.csv", "G", shank, paw, harmonics=5
    )
    rows = []
    for ref in strides:
        for condition, policy in [
            ("static_single_device", common_static),
            ("adaptive_impedance", adaptive_policy[ref.gait_type]),
        ]:
            d = simulate(ref, policy, mech, steps_per_cycle=360, samples_per_cycle=201, settling_cycles=4)
            rows.append({
                "stride_id": ref.source_id, "gait_type": ref.gait_type,
                "condition": condition, **metrics(d, ref, mech),
            })
    out = pd.DataFrame(rows)
    out.to_csv(CSV / "heldout_stride_validation.csv", index=False)
    paired = out.pivot(index=["stride_id", "gait_type"], columns="condition", values="objective").reset_index()
    paired["adaptive_improvement_percent"] = 100.0 * (
        paired["static_single_device"] - paired["adaptive_impedance"]
    ) / paired["static_single_device"]
    paired.to_csv(CSV / "heldout_stride_paired_improvement.csv", index=False)
    return out, paired


def robustness_and_sensitivity(refs, mech, common_static, adaptive_pars):
    phase_errors = pd.read_csv(DATA / "phase_timing_error_summary.csv")
    delay_rows, sensitivity_rows, conv_rows = [], [], []
    for gait in ("walk", "trot"):
        observed = float(phase_errors[phase_errors.true_gait_type.eq(gait)]["start_time_mae_s"].mean())
        for label, delay in [("ideal", 0.0), ("observed_MAE", observed), ("double_observed_MAE", 2.0 * observed)]:
            p = build_adaptive_policy(refs[gait], mech, adaptive_pars[gait], sensor_delay_s=delay)
            d = simulate(refs[gait], p, mech, 540, 301, 5)
            delay_rows.append({"gait_type": gait, "delay_case": label, "sensor_delay_s": delay, **metrics(d, refs[gait], mech)})

        base = adaptive_pars[gait]
        for parameter in ("k_stance", "k_swing", "compression_target_m", "feedforward_gain"):
            for scale in (0.9, 1.0, 1.1):
                pset = dict(base)
                pset[parameter] = base[parameter] * scale
                p = build_adaptive_policy(refs[gait], mech, pset)
                d = simulate(refs[gait], p, mech, 360, 201, 4)
                sensitivity_rows.append({
                    "gait_type": gait, "parameter": parameter, "scale": scale,
                    "value": pset[parameter], **metrics(d, refs[gait], mech),
                })

        for condition, policy in [
            ("static_single_device", common_static),
            ("adaptive_impedance", build_adaptive_policy(refs[gait], mech, adaptive_pars[gait])),
        ]:
            for steps in (240, 480, 720, 1200):
                d = simulate(refs[gait], policy, mech, steps, 301, 5)
                conv_rows.append({
                    "gait_type": gait, "condition": condition, "steps_per_cycle": steps,
                    **metrics(d, refs[gait], mech),
                })
    delay = pd.DataFrame(delay_rows); delay.to_csv(CSV / "phase_delay_robustness.csv", index=False)
    sensitivity = pd.DataFrame(sensitivity_rows); sensitivity.to_csv(CSV / "local_sensitivity.csv", index=False)
    convergence = pd.DataFrame(conv_rows); convergence.to_csv(CSV / "numerical_convergence.csv", index=False)
    return delay, sensitivity, convergence


def dataset_inventory():
    strides = pd.read_csv(DATA / "Canine_Prosthesis_Kinematic_Stride_Summary.csv")
    report = pd.read_csv(DATA / "four_state_classification_report.csv")
    rows = [
        {"source": "Kinematic_Targets_101", "role": "Dog G mean reference and held-out stride validation", "records": len(pd.read_csv(DATA / "Canine_Prosthesis_Kinematic_Targets_101.csv")), "notes": "Processed from uploaded MAT motion files"},
        {"source": "Kinematic_Stride_Summary", "role": "population plausibility ranges", "records": len(strides), "notes": f"{strides.dog_id.nunique()} dogs; {strides.gait_type.nunique()} gait types"},
        {"source": "DogG_personalized_reference", "role": "controller design target", "records": len(pd.read_csv(DATA / "dogG_personalized_reference_101.csv")), "notes": "7 walk and 6 trot strides"},
        {"source": "phase_timing_error_summary", "role": "sensor-delay robustness", "records": len(pd.read_csv(DATA / "phase_timing_error_summary.csv")), "notes": "Observed phase-start timing error"},
        {"source": "four_state_classification_report", "role": "deployment context only", "records": len(report), "notes": "Not used as proof of a trained raw IMU+FSR CNN"},
    ]
    pd.DataFrame(rows).to_csv(CSV / "dataset_inventory.csv", index=False)


def parameter_dict(mech, common_static, gait_static, adaptive_pars):
    return {
        "model_version": "3.0-variable-impedance",
        "subject": {"dog_id": "G", "breed": "Labrador Retriever", "body_mass_kg": mech.body_mass_kg},
        "mechanical_parameters": mech.__dict__,
        "static_single_device": common_static.__dict__,
        "static_gait_specific": {g: p.__dict__ for g, p in gait_static.items()},
        "adaptive_parameters": adaptive_pars,
        "controller": {
            "type": "phase-continuous computed-equilibrium variable impedance",
            "equation": "tau = -K(phi)*(theta-theta_eq(phi)) - B(phi)*theta_dot",
            "controlled_terms": ["K", "B", "theta_eq", "k_paw"],
            "actuator_lags_included": True,
            "torque_saturation_included": True,
        },
    }


def make_plots(sims, summary, improvements, validation, delay):
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {"natural": "#222222", "static": "#d95f02", "adaptive": "#1b9e77"}
    for gait in ("walk", "trot"):
        st = sims[(gait, "static_single_device")]
        ad = sims[(gait, "adaptive_impedance")]
        fig, ax = plt.subplots(figsize=(9.2, 5.2))
        ax.plot(st.cycle_percent, np.rad2deg(st.hock_angle_rad_ref), lw=2.6, color=colors["natural"], label="Natural reference")
        ax.plot(st.cycle_percent, np.rad2deg(st.hock_angle_rad_sim), lw=2.0, color=colors["static"], label="Single static device")
        ax.plot(ad.cycle_percent, np.rad2deg(ad.hock_angle_rad_sim), lw=2.2, color=colors["adaptive"], label="Adaptive impedance")
        ax.axvline(100 * refs_global[gait].stance_fraction, color="#777777", ls="--", lw=1)
        ax.set(xlabel="Gait cycle (%)", ylabel="Hock angle (deg)", title=f"Dog G {gait}: hock-angle tracking")
        ax.legend(); fig.tight_layout(); fig.savefig(PLOTS / f"01_{gait}_hock_angle.png", dpi=220); plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.2, 6.0))
        ax.plot(st.toe_x_m_ref, st.toe_y_m_ref, lw=2.6, color=colors["natural"], label="Natural reference")
        ax.plot(st.toe_x_m_sim, st.toe_y_m_sim, lw=2.0, color=colors["static"], label="Single static device")
        ax.plot(ad.toe_x_m_sim, ad.toe_y_m_sim, lw=2.2, color=colors["adaptive"], label="Adaptive impedance")
        ax.set(xlabel="Toe x (m)", ylabel="Toe y (m)", title=f"Dog G {gait}: toe trajectory")
        ax.axis("equal"); ax.legend(); fig.tight_layout(); fig.savefig(PLOTS / f"02_{gait}_toe_path.png", dpi=220); plt.close(fig)

        fig, axes = plt.subplots(2, 1, figsize=(9.2, 7.0), sharex=True)
        bw = 30.0 * 9.81
        axes[0].plot(st.cycle_percent, st.normal_force_N / bw, color=colors["static"], label="Static")
        axes[0].plot(ad.cycle_percent, ad.normal_force_N / bw, color=colors["adaptive"], label="Adaptive")
        axes[0].plot(ad.cycle_percent, ad.target_load_N / bw, color="#222222", ls="--", label="Load target")
        axes[0].set_ylabel("Vertical load (BW)"); axes[0].legend(ncol=3)
        axes[1].plot(st.cycle_percent, st.paw_compression_m * 1000, color=colors["static"], label="Static")
        axes[1].plot(ad.cycle_percent, ad.paw_compression_m * 1000, color=colors["adaptive"], label="Adaptive")
        axes[1].set(xlabel="Gait cycle (%)", ylabel="Compression (mm)")
        fig.suptitle(f"Dog G {gait}: loading and paw compliance"); fig.tight_layout(); fig.savefig(PLOTS / f"03_{gait}_loading.png", dpi=220); plt.close(fig)

        fig, axes = plt.subplots(4, 1, figsize=(9.2, 9.2), sharex=True)
        axes[0].plot(ad.cycle_percent, ad.k_hock_command_Nm_rad); axes[0].set_ylabel("K (N m/rad)")
        axes[1].plot(ad.cycle_percent, ad.b_hock_command_Nm_s_rad); axes[1].set_ylabel("B (N m s/rad)")
        axes[2].plot(ad.cycle_percent, np.rad2deg(ad.theta_eq_command_rad)); axes[2].set_ylabel("theta_eq (deg)")
        axes[3].plot(ad.cycle_percent, ad.k_paw_command_N_m / 1000); axes[3].set_ylabel("k_paw (kN/m)"); axes[3].set_xlabel("Gait cycle (%)")
        fig.suptitle(f"Dog G {gait}: adaptive impedance commands"); fig.tight_layout(); fig.savefig(PLOTS / f"04_{gait}_controller_profiles.png", dpi=220); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    x = np.arange(2)
    for condition, label, color in [
        ("static_single_device", "Single static", colors["static"]),
        ("static_gait_specific", "Gait-specific static", "#7570b3"),
        ("adaptive_impedance", "Adaptive", colors["adaptive"]),
    ]:
        vals = [float(summary[(summary.gait_type == g) & (summary.condition == condition)].objective.iloc[0]) for g in ("walk", "trot")]
        offset = {"static_single_device": -0.24, "static_gait_specific": 0.0, "adaptive_impedance": 0.24}[condition]
        axes[0].bar(x + offset, vals, width=0.22, label=label, color=color)
    axes[0].set_xticks(x, ["Walk", "Trot"]); axes[0].set_ylabel("Objective (lower is better)"); axes[0].legend(fontsize=8)
    axes[1].bar(x, improvements.objective_improvement_vs_single_static_percent, color=colors["adaptive"])
    axes[1].set_xticks(x, ["Walk", "Trot"]); axes[1].set_ylabel("Improvement vs single static (%)")
    fig.suptitle("Final model comparison"); fig.tight_layout(); fig.savefig(PLOTS / "05_summary_comparison.png", dpi=220); plt.close(fig)

    paired = validation.pivot_table(index=["stride_id", "gait_type"], columns="condition", values="objective").reset_index()
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    for gait, marker in [("walk", "o"), ("trot", "s")]:
        g = paired[paired.gait_type.eq(gait)]
        ax.scatter(g.static_single_device, g.adaptive_impedance, label=gait.title(), s=55, marker=marker)
    lim = [0, max(paired.static_single_device.max(), paired.adaptive_impedance.max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=1); ax.set(xlim=lim, ylim=lim, xlabel="Static objective", ylabel="Adaptive objective", title="Held-out individual strides")
    ax.legend(); fig.tight_layout(); fig.savefig(PLOTS / "06_heldout_stride_validation.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for gait in ("walk", "trot"):
        g = delay[delay.gait_type.eq(gait)].sort_values("sensor_delay_s")
        ax.plot(g.sensor_delay_s * 1000, g.objective, marker="o", label=gait.title())
    ax.set(xlabel="Simulated phase delay (ms)", ylabel="Adaptive objective", title="Robustness to observed sensor timing error")
    ax.legend(); fig.tight_layout(); fig.savefig(PLOTS / "07_phase_delay_robustness.png", dpi=220); plt.close(fig)


def make_video(gait, ref, static_df, adaptive_df, output):
    phases = np.linspace(0.0, 99.5, 120)
    with tempfile.TemporaryDirectory(prefix=f"canine_{gait}_frames_") as td:
        td = Path(td)
        all_x = np.r_[static_df.hock_x_m, static_df.toe_x_m_ref, static_df.toe_x_m_sim, adaptive_df.toe_x_m_sim]
        all_y = np.r_[static_df.hock_y_m, static_df.toe_y_m_ref, static_df.toe_y_m_sim, adaptive_df.toe_y_m_sim]
        padx = 0.12 * max(np.ptp(all_x), 0.2); pady = 0.20 * max(np.ptp(all_y), 0.15)
        xlim = (all_x.min() - padx, all_x.max() + padx); ylim = (all_y.min() - pady, all_y.max() + pady)
        for fi, pct in enumerate(phases):
            i = int(np.argmin(np.abs(static_df.cycle_percent.to_numpy() - pct)))
            j = int(np.argmin(np.abs(adaptive_df.cycle_percent.to_numpy() - pct)))
            fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
            gs = fig.add_gridspec(2, 3, height_ratios=[2.4, 1.0])
            modes = [
                ("Natural", static_df.toe_x_m_ref, static_df.toe_y_m_ref, "#222222", i),
                ("Static", static_df.toe_x_m_sim, static_df.toe_y_m_sim, "#d95f02", i),
                ("Adaptive", adaptive_df.toe_x_m_sim, adaptive_df.toe_y_m_sim, "#1b9e77", j),
            ]
            for col, (label, tx, ty, color, idx) in enumerate(modes):
                ax = fig.add_subplot(gs[0, col])
                hx = float(static_df.hock_x_m.iloc[i]); hy = float(static_df.hock_y_m.iloc[i])
                knee = ref.evaluate(static_df.time_s.iloc[i])
                kx, ky = float(knee["knee_x"]), float(knee["knee_y"])
                ax.plot(static_df.toe_x_m_ref, static_df.toe_y_m_ref, color="#bbbbbb", lw=1.0)
                ax.plot(tx.iloc[: idx + 1], ty.iloc[: idx + 1], color=color, lw=2.0, alpha=0.8)
                ax.plot([kx, hx, float(tx.iloc[idx])], [ky, hy, float(ty.iloc[idx])], "-o", color=color, lw=4, ms=7)
                ax.set(xlim=xlim, ylim=ylim, aspect="equal", title=label, xlabel="x (m)")
                if col == 0: ax.set_ylabel("y (m)")
                if label == "Adaptive":
                    ax.text(0.03, 0.03, f"K={adaptive_df.k_hock_effective_Nm_rad.iloc[j]:.0f} N m/rad\nB={adaptive_df.b_hock_effective_Nm_s_rad.iloc[j]:.2f} N m s/rad\nk_paw={adaptive_df.k_paw_effective_N_m.iloc[j]/1000:.1f} kN/m", transform=ax.transAxes, fontsize=9)
            ax = fig.add_subplot(gs[1, :])
            ax.plot(static_df.cycle_percent, np.rad2deg(static_df.hock_angle_rad_ref), color="#222222", label="Natural")
            ax.plot(static_df.cycle_percent, np.rad2deg(static_df.hock_angle_rad_sim), color="#d95f02", label="Static")
            ax.plot(adaptive_df.cycle_percent, np.rad2deg(adaptive_df.hock_angle_rad_sim), color="#1b9e77", label="Adaptive")
            ax.axvline(pct, color="#555555", lw=1.5); ax.set(xlim=(0, 100), xlabel="Gait cycle (%)", ylabel="Hock angle (deg)")
            ax.legend(ncol=3, loc="upper right")
            phase_name = "stance" if pct < 100 * ref.stance_fraction else "swing"
            fig.suptitle(f"Dog G {gait.title()} | Natural vs Static vs Adaptive | {phase_name}", fontsize=15)
            fig.tight_layout()
            fig.savefig(td / f"frame_{fi:04d}.png")
            plt.close(fig)
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", "30", "-i", str(td / "frame_%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(output)
        ], check=True)


def write_checks_and_docs(mech, summary, improvements, validation, delay, convergence, parameters):
    checks = []
    for _, r in summary.iterrows():
        checks.extend([
            {"gait_type": r.gait_type, "condition": r.condition, "check": "peak_hock_moment_within_limit", "value": r.peak_hock_moment_Nm_per_kg, "limit": mech.torque_limit_Nm_per_kg, "pass": r.peak_hock_moment_Nm_per_kg <= mech.torque_limit_Nm_per_kg + 1e-9},
            {"gait_type": r.gait_type, "condition": r.condition, "check": "paw_compression_below_25_mm", "value": r.max_paw_compression_mm, "limit": 25.0, "pass": r.max_paw_compression_mm <= 25.0},
            {"gait_type": r.gait_type, "condition": r.condition, "check": "torque_saturation_below_2_percent", "value": r.torque_saturation_fraction, "limit": 0.02, "pass": r.torque_saturation_fraction <= 0.02},
        ])
    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(CSV / "physics_checks.csv", index=False)
    (ROOT / "config" / "final_parameters.json").write_text(json.dumps(parameters, indent=2), encoding="utf-8")

    lines = [
        "# Final Results — Variable-Impedance Canine Prosthesis v3.0", "",
        "## Scope", "",
        "Subject-specific 2D sagittal simulation for Dog G. The natural knee is preserved. The model controls the distal hock/paw using phase-continuous equivalent impedance.", "",
        "## Final comparison", "",
    ]
    for _, r in improvements.iterrows():
        lines += [
            f"### {r.gait_type.title()}", "",
            f"- Objective improvement vs one fixed device: **{r.objective_improvement_vs_single_static_percent:.1f}%**",
            f"- Objective improvement vs gait-specific optimized static: **{r.objective_improvement_vs_gait_specific_static_percent:.1f}%**",
            f"- Hock-angle RMSE reduction vs one fixed device: **{r.angle_rmse_reduction_vs_single_static_percent:.1f}%**",
            f"- Toe-position RMSE reduction vs one fixed device: **{r.toe_rmse_reduction_vs_single_static_percent:.1f}%**", "",
        ]
    paired_mean = validation.groupby("gait_type")["adaptive_improvement_percent"].agg(["mean", "median", "min"]).reset_index()
    lines += ["## Held-out individual strides", ""]
    for _, r in paired_mean.iterrows():
        lines.append(f"- {r.gait_type.title()}: mean {r['mean']:.1f}%, median {r['median']:.1f}%, worst stride {r['min']:.1f}% improvement.")
    lines += [
        "", "## Interpretation", "",
        "The visible difference comes from adding controlled equilibrium angle and damping to stiffness adaptation. This is a variable-impedance simulation, not a claim that a passive spring alone can reproduce the natural trajectory.", "",
        "The adaptive profile is trained on the Dog G mean reference and tested on individual Dog G strides. Raw synchronized 6-axis IMU + FSR windows were not available, so the included TCN remains a deployment template. Existing phase-timing errors are used only for delay robustness.", "",
        "## Fixed limitations", "",
        "- Proximal knee-to-hock motion is prescribed from vision; whole-body dynamics are not solved.",
        "- Vertical loading uses literature-calibrated targets rather than synchronized Dog G force-plate measurements.",
        "- Work values are simulation proxies, not battery or metabolic energy.",
        "- Equivalent/optimal stiffness is reported; no claim of true biological stiffness is made.",
    ]
    (ROOT / "FINAL_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    global refs_global
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Reduced optimization for smoke tests")
    parser.add_argument("--skip-video", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    shank, paw = geometry_from_data()
    mech = MechanicalParameters(paw_length_m=paw)
    refs = load_mean_refs(mech); refs_global = refs
    dataset_inventory()
    common_static, gait_static, adaptive_pars, adaptive_policy = optimize_policies(refs, mech, args.quick)
    parameters = parameter_dict(mech, common_static, gait_static, adaptive_pars)
    sims, summary, improvements = evaluate_final(refs, mech, common_static, gait_static, adaptive_policy)
    validation_raw, validation = heldout_validation(mech, refs, common_static, adaptive_policy)
    delay, sensitivity, convergence = robustness_and_sensitivity(refs, mech, common_static, adaptive_pars)
    make_plots(sims, summary, improvements, validation_raw, delay)
    write_checks_and_docs(mech, summary, improvements, validation, delay, convergence, parameters)
    if not args.skip_video:
        for gait in ("walk", "trot"):
            make_video(gait, refs[gait], sims[(gait, "static_single_device")], sims[(gait, "adaptive_impedance")], VIDEOS / f"DogG_{gait}_natural_static_adaptive.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(VIDEOS / "DogG_walk_natural_static_adaptive.mp4"),
            "-i", str(VIDEOS / "DogG_trot_natural_static_adaptive.mp4"),
            "-filter_complex", "hstack=inputs=2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21",
            str(VIDEOS / "DogG_walk_trot_natural_static_adaptive.mp4")
        ], check=True)
    print(improvements.to_string(index=False))
    print("\nPhysics checks:")
    print(pd.read_csv(CSV / "physics_checks.csv").groupby("pass").size().to_string())


if __name__ == "__main__":
    main()
