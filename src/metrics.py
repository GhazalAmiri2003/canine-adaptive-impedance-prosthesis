from __future__ import annotations

import numpy as np
import pandas as pd
from model import GRF_TARGETS, MechanicalParameters
from reference import GaitReference, wrap_pi


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, float) ** 2)))


def metrics(df: pd.DataFrame, ref: GaitReference, mech: MechanicalParameters) -> dict:
    t = df["time_s"].to_numpy(float)
    angle_error = wrap_pi(df["hock_angle_rad_sim"] - df["hock_angle_rad_ref"])
    dx = df["toe_x_m_sim"] - df["toe_x_m_ref"]
    dy = df["toe_y_m_sim"] - df["toe_y_m_ref"]
    dvx = df["toe_vx_m_s_sim"] - df["toe_vx_m_s_ref"]
    dvy = df["toe_vy_m_s_sim"] - df["toe_vy_m_s_ref"]
    dax = df["toe_ax_m_s2_sim"] - df["toe_ax_m_s2_ref"]
    day = df["toe_ay_m_s2_sim"] - df["toe_ay_m_s2_ref"]
    jerk_sim = np.column_stack([
        np.gradient(df["toe_ax_m_s2_sim"], t), np.gradient(df["toe_ay_m_s2_sim"], t)
    ])
    jerk_ref = np.column_stack([
        np.gradient(df["toe_ax_m_s2_ref"], t), np.gradient(df["toe_ay_m_s2_ref"], t)
    ])

    angle_scale = max(float(np.ptp(df["hock_angle_rad_ref"])), np.deg2rad(10.0))
    toe_scale = max(float(np.ptp(df["toe_x_m_ref"])), 0.05)
    vel_scale = max(float(np.ptp(df["toe_vx_m_s_ref"])), float(np.ptp(df["toe_vy_m_s_ref"])), 0.2)
    acc_scale = max(float(np.ptp(df["toe_ax_m_s2_ref"])), float(np.ptp(df["toe_ay_m_s2_ref"])), 1.0)
    jerk_scale = max(float(np.ptp(jerk_ref[:, 0])), float(np.ptp(jerk_ref[:, 1])), 1.0)

    target = GRF_TARGETS[ref.gait_type]
    bw = mech.body_mass_kg * mech.gravity_m_s2
    peak_BW = float(df["normal_force_N"].max() / bw)
    impulse = float(np.trapezoid(df["normal_force_N"], t) / bw)
    peak_error = abs(peak_BW - target["peak_BW"]) / target["peak_BW"]
    impulse_error = abs(impulse - target["impulse_BW_s"]) / target["impulse_BW_s"]
    max_comp = float(df["paw_compression_m"].max())
    peak_tau_norm = float(df["tau_hock_Nm"].abs().max() / mech.body_mass_kg)
    tuning_work = float(np.trapezoid(df["equilibrium_tuning_power_W"], t))
    positive_joint_work = float(np.trapezoid(np.maximum(df["joint_power_W"], 0.0), t))
    sat_fraction = float(df["torque_saturated"].mean())

    angle_nrmse = rms(angle_error) / angle_scale
    pos_nrmse = rms(np.hypot(dx, dy)) / toe_scale
    vel_nrmse = rms(np.hypot(dvx, dvy)) / vel_scale
    acc_nrmse = rms(np.hypot(dax, day)) / acc_scale
    jerk_nrmse = rms(np.linalg.norm(jerk_sim - jerk_ref, axis=1)) / jerk_scale
    compression_pen = max(0.0, max_comp / 0.025 - 1.0) ** 2
    torque_pen = max(0.0, peak_tau_norm / mech.torque_limit_Nm_per_kg - 1.0) ** 2
    # Continuous energy regularization prevents trajectory tracking from being
    # improved merely by demanding an unnecessarily aggressive neutral-angle actuator.
    tuning_pen = (tuning_work / 15.0) ** 2
    cost = (
        0.38 * angle_nrmse + 0.27 * pos_nrmse + 0.10 * vel_nrmse
        + 0.06 * acc_nrmse + 0.04 * jerk_nrmse
        + 0.06 * peak_error + 0.03 * impulse_error
        + 0.025 * compression_pen + 0.025 * torque_pen + 0.025 * tuning_pen
    )
    return {
        "objective": float(cost),
        "hock_angle_rmse_deg": float(np.rad2deg(rms(angle_error))),
        "hock_angle_nrmse": angle_nrmse,
        "toe_position_rmse_mm": float(rms(np.hypot(dx, dy)) * 1000.0),
        "toe_position_nrmse": pos_nrmse,
        "toe_velocity_rmse_m_s": rms(np.hypot(dvx, dvy)),
        "toe_velocity_nrmse": vel_nrmse,
        "toe_acceleration_rmse_m_s2": rms(np.hypot(dax, day)),
        "toe_acceleration_nrmse": acc_nrmse,
        "jerk_tracking_nrmse": jerk_nrmse,
        "peak_vertical_load_BW": peak_BW,
        "target_peak_vertical_load_BW": target["peak_BW"],
        "peak_load_relative_error": peak_error,
        "vertical_impulse_BW_s": impulse,
        "target_vertical_impulse_BW_s": target["impulse_BW_s"],
        "impulse_relative_error": impulse_error,
        "max_paw_compression_mm": max_comp * 1000.0,
        "peak_hock_moment_Nm_per_kg": peak_tau_norm,
        "torque_saturation_fraction": sat_fraction,
        "equilibrium_tuning_work_proxy_J": tuning_work,
        "positive_joint_work_proxy_J": positive_joint_work,
    }


def improvement_percent(static_value, adaptive_value):
    return 100.0 * (float(static_value) - float(adaptive_value)) / max(abs(float(static_value)), 1e-12)
