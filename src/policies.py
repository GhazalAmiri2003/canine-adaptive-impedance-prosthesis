from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d

from model import MechanicalParameters, calibrate_load_profile, target_load_N
from reference import GaitReference


@dataclass(frozen=True)
class StaticImpedancePolicy:
    k_hock: float
    b_hock: float
    theta_eq: float
    k_paw: float

    def command(self, t, ref):
        return {
            "k_hock": self.k_hock,
            "b_hock": self.b_hock,
            "theta_eq": self.theta_eq,
            "k_paw": self.k_paw,
        }


class PeriodicProfile:
    def __init__(self, phase: np.ndarray, values: np.ndarray):
        values = np.asarray(values, float).copy()
        values[-1] = values[0]
        self.spline = CubicSpline(np.asarray(phase, float), values, bc_type="periodic")

    def __call__(self, phase):
        return self.spline(np.mod(phase, 1.0))


class AdaptiveImpedancePolicy:
    def __init__(self, phase, k, b, theta_eq, k_paw, lead_fraction=0.0, sensor_delay_s=0.0, name="adaptive"):
        self.k = PeriodicProfile(phase, k)
        self.b = PeriodicProfile(phase, b)
        self.theta_eq = PeriodicProfile(phase, np.unwrap(theta_eq))
        self.k_paw = PeriodicProfile(phase, k_paw)
        self.lead_fraction = float(lead_fraction)
        self.sensor_delay_s = float(sensor_delay_s)
        self.name = name

    def command(self, t, ref):
        phase = float(ref.phase_fraction(t)) + self.lead_fraction - self.sensor_delay_s / ref.T
        return {
            "k_hock": float(self.k(phase)),
            "b_hock": float(self.b(phase)),
            "theta_eq": float(self.theta_eq(phase)),
            "k_paw": float(self.k_paw(phase)),
        }


def build_adaptive_policy(ref: GaitReference, mech: MechanicalParameters, pars: dict, sensor_delay_s: float = 0.0):
    """Create a smooth phase-continuous computed-equilibrium impedance profile."""
    n = 401
    phase = np.linspace(0.0, 1.0, n)
    t = phase * ref.T
    r = ref.evaluate(t)
    stance = (phase < ref.stance_fraction).astype(float)
    stance[-1] = stance[0]
    stance_soft = gaussian_filter1d(stance[:-1], sigma=5.0, mode="wrap")
    stance_soft = np.r_[stance_soft, stance_soft[0]]

    k = pars["k_swing"] + (pars["k_stance"] - pars["k_swing"]) * stance_soft
    zeta = pars["zeta_swing"] + (pars["zeta_stance"] - pars["zeta_swing"]) * stance_soft
    b = np.clip(2.0 * zeta * np.sqrt(k * mech.J_hock), 0.05, 1.50)

    theta_ref = np.unwrap(np.asarray(r["theta_ref"], float))
    center = float(np.mean(theta_ref))
    theta_target = center + pars["theta_scale"] * (theta_ref - center)
    theta_target[-1] = theta_target[0]
    theta_spline = CubicSpline(t, theta_target, bc_type="periodic")
    theta_dot = theta_spline(t, 1)
    q_target = np.asarray(r["phi_prox"], float) + theta_target
    q_target[-1] = q_target[0]
    q_spline = CubicSpline(t, q_target, bc_type="periodic")
    qdd = q_spline(t, 2)

    calibration = calibrate_load_profile(ref, mech)
    load = target_load_N(t, ref, mech, calibration)
    et = np.column_stack([-np.sin(q_target), np.cos(q_target)])
    hock_a = np.column_stack([r["hock_ax"], r["hock_ay"]])
    tau_gravity = -mech.paw_mass_kg * mech.gravity_m_s2 * mech.paw_com_m * np.cos(q_target)
    tau_base = -mech.paw_mass_kg * mech.paw_com_m * np.sum(hock_a * et, axis=1)
    tau_contact = mech.cop_fraction_paw * mech.paw_length_m * np.cos(q_target) * load
    tau_required = mech.J_hock * qdd - tau_gravity - tau_base - tau_contact
    tau_required = np.clip(tau_required, -mech.torque_limit_Nm, mech.torque_limit_Nm)

    theta_eq = theta_target + pars["feedforward_gain"] * (tau_required + b * theta_dot) / k
    max_offset = np.deg2rad(28.0)
    theta_eq = np.clip(theta_eq, theta_target - max_offset, theta_target + max_offset)
    theta_eq = gaussian_filter1d(theta_eq[:-1], sigma=7.0, mode="wrap")
    theta_eq = np.r_[theta_eq, theta_eq[0]]

    delta_target = pars["compression_target_m"]
    kp_raw = np.where(load > 1.0, load / max(delta_target, 1e-4), pars["k_paw_preload"])
    kp_raw = np.clip(kp_raw, 6000.0, 50000.0)
    kp = gaussian_filter1d(kp_raw[:-1], sigma=7.0, mode="wrap")
    kp = np.r_[kp, kp[0]]

    return AdaptiveImpedancePolicy(
        phase, k, b, theta_eq, kp,
        lead_fraction=pars["lead_fraction"], sensor_delay_s=sensor_delay_s,
    )
