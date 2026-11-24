from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from reference import GaitReference, wrap_pi


@dataclass(frozen=True)
class MechanicalParameters:
    body_mass_kg: float = 30.0
    paw_length_m: float = 0.3233306369
    paw_mass_kg: float = 0.21
    paw_com_fraction: float = 0.47
    paw_I_cm_kg_m2: float = 0.000609
    gravity_m_s2: float = 9.81
    contact_damping_N_s_m: float = 500.0
    # Effective sagittal COP lever calibrated so the natural-reference inverse
    # dynamics stays near the published Labrador hock-moment range.
    cop_fraction_paw: float = 0.22
    torque_limit_Nm_per_kg: float = 0.45
    tau_k_s: float = 0.045
    tau_b_s: float = 0.045
    tau_theta_eq_s: float = 0.025
    tau_k_paw_s: float = 0.070
    theta_eq_rate_limit_rad_s: float = 12.0

    @property
    def paw_com_m(self):
        return self.paw_com_fraction * self.paw_length_m

    @property
    def J_hock(self):
        return self.paw_I_cm_kg_m2 + self.paw_mass_kg * self.paw_com_m**2

    @property
    def torque_limit_Nm(self):
        return self.torque_limit_Nm_per_kg * self.body_mass_kg


GRF_TARGETS = {
    "walk": {"peak_BW": 0.3866, "impulse_BW_s": 0.1396},
    "trot": {"peak_BW": 0.6450, "impulse_BW_s": 0.0910},
}


def calibrate_load_profile(ref: GaitReference, mech: MechanicalParameters):
    """Match published peak and impulse with a fixed acceleration-shaped demand."""
    from scipy.optimize import brentq

    target = GRF_TARGETS[ref.gait_type]
    stance_time = ref.stance_fraction * ref.T
    mean_target = target["impulse_BW_s"] / stance_time
    ratio_target = target["peak_BW"] / max(mean_target, 1e-12)
    t = np.linspace(0.0, stance_time, 2001, endpoint=False)
    a_norm = np.asarray(ref.hock_y(t, 2), float) / mech.gravity_m_s2

    def shape(beta):
        return np.maximum(0.0, 1.0 + beta * a_norm)

    def error(beta):
        s = shape(beta)
        return float(np.max(s) / np.mean(s) - ratio_target)

    if ratio_target <= 1.000001:
        beta = 0.0
    else:
        hi = 0.5
        while error(hi) < 0.0 and hi < 32.0:
            hi *= 2.0
        beta = float(brentq(error, 0.0, hi))
    s = shape(beta)
    support_fraction = mean_target / max(float(np.mean(s)), 1e-12)
    return support_fraction, beta


def target_load_N(t, ref: GaitReference, mech: MechanicalParameters, calibration):
    support_fraction, beta = calibration
    r = ref.evaluate(t)
    stance = np.asarray(ref.phase_fraction(t) < ref.stance_fraction, float)
    return (
        mech.body_mass_kg * support_fraction
        * np.maximum(0.0, mech.gravity_m_s2 + beta * np.asarray(r["hock_ay"], float))
        * stance
    )


def _rk4(fun, t, y, dt):
    k1 = fun(t, y)
    k2 = fun(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = fun(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = fun(t + dt, y + dt * k3)
    return y + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


class ImpedanceDynamics:
    """Reduced-order hock/paw dynamics with rate-limited variable impedance."""

    def __init__(self, ref: GaitReference, mech: MechanicalParameters, policy):
        self.ref = ref
        self.m = mech
        self.policy = policy
        self.calibration = calibrate_load_profile(ref, mech)

    def diagnostics(self, t, y):
        q, qdot, delta, deltadot, k_eff, b_eff, theta_eq_eff, k_paw_eff = y
        r = self.ref.evaluate(t)
        cmd = self.policy.command(t, self.ref)

        dk = (cmd["k_hock"] - k_eff) / self.m.tau_k_s
        db = (cmd["b_hock"] - b_eff) / self.m.tau_b_s
        dtheta_eq = np.clip(
            wrap_pi(cmd["theta_eq"] - theta_eq_eff) / self.m.tau_theta_eq_s,
            -self.m.theta_eq_rate_limit_rad_s,
            self.m.theta_eq_rate_limit_rad_s,
        )
        dkp = (cmd["k_paw"] - k_paw_eff) / self.m.tau_k_paw_s

        phase = self.ref.phase(t)
        load = float(target_load_N(t, self.ref, self.m, self.calibration))
        delta_pos = max(0.0, float(delta))
        internal_force = max(0.0, k_paw_eff * delta_pos + self.m.contact_damping_N_s_m * deltadot)
        if phase == "stance":
            deltaddot = (load - internal_force) / max(self.m.body_mass_kg * self.calibration[0], 1e-6)
            normal_force = internal_force
        else:
            deltaddot = -internal_force / max(self.m.body_mass_kg * self.calibration[0], 1e-6)
            normal_force = 0.0

        theta = float(wrap_pi(q - float(r["phi_prox"])))
        theta_dot = qdot - float(r["phidot_prox"])
        err = float(wrap_pi(theta - theta_eq_eff))
        tau_spring = -k_eff * err
        tau_damping = -b_eff * theta_dot
        tau_raw = tau_spring + tau_damping
        tau_hock = float(np.clip(tau_raw, -self.m.torque_limit_Nm, self.m.torque_limit_Nm))

        a = self.m.paw_com_m
        L = self.m.paw_length_m
        J = self.m.J_hock
        et = np.array([-np.sin(q), np.cos(q)])
        hock_a = np.array([r["hock_ax"], r["hock_ay"]], float)
        tau_gravity = -self.m.paw_mass_kg * self.m.gravity_m_s2 * a * np.cos(q)
        tau_base = -self.m.paw_mass_kg * a * float(np.dot(hock_a, et))
        tau_contact = self.m.cop_fraction_paw * L * np.cos(q) * normal_force
        qddot = (tau_hock + tau_gravity + tau_base + tau_contact) / J

        hock = np.array([r["hock_x"], r["hock_y"]], float)
        hock_v = np.array([r["hock_vx"], r["hock_vy"]], float)
        er = np.array([np.cos(q), np.sin(q)])
        toe = hock + L * er
        toe_v = hock_v + L * qdot * et

        return {
            "qddot": qddot, "deltaddot": deltaddot,
            "dk": dk, "db": db, "dtheta_eq": dtheta_eq, "dkp": dkp,
            "phase": phase, "theta": theta, "theta_dot": theta_dot,
            "theta_error": err, "hock_x": hock[0], "hock_y": hock[1],
            "toe_x": toe[0], "toe_y": toe[1], "toe_vx": toe_v[0], "toe_vy": toe_v[1],
            "normal_force": normal_force, "target_load": load,
            "compression": delta_pos, "tau_hock": tau_hock, "tau_hock_raw": tau_raw,
            "tau_spring": tau_spring, "tau_damping": tau_damping,
            "tau_contact": tau_contact, "tau_base": tau_base, "tau_gravity": tau_gravity,
            "torque_saturated": float(abs(tau_raw) > self.m.torque_limit_Nm + 1e-9),
            **{f"cmd_{k}": v for k, v in cmd.items()},
        }

    def rhs(self, t, y):
        d = self.diagnostics(t, y)
        return np.array([
            y[1], d["qddot"], y[3], d["deltaddot"],
            d["dk"], d["db"], d["dtheta_eq"], d["dkp"],
        ], float)


def simulate(
    ref: GaitReference,
    policy,
    mech: MechanicalParameters,
    steps_per_cycle: int = 480,
    samples_per_cycle: int = 301,
    settling_cycles: int = 4,
) -> pd.DataFrame:
    dyn = ImpedanceDynamics(ref, mech, policy)
    c0 = policy.command(0.0, ref)
    q0 = float(ref.q_ref(0.0))
    qdot0 = float(ref.q_ref(0.0, 1))
    load0 = float(target_load_N(0.0, ref, mech, dyn.calibration))
    delta0 = load0 / max(c0["k_paw"], 1.0)
    y = np.array([q0, qdot0, delta0, 0.0, c0["k_hock"], c0["b_hock"], c0["theta_eq"], c0["k_paw"]])
    dt = ref.T / int(steps_per_cycle)
    times, states = [], []
    total = (settling_cycles + 1) * int(steps_per_cycle)
    for n in range(total):
        t = n * dt
        y = _rk4(dyn.rhs, t, y, dt)
        if y[2] < 0.0:
            y[2] = 0.0
            y[3] = max(0.0, y[3])
        if n >= settling_cycles * int(steps_per_cycle):
            times.append((n + 1) * dt)
            states.append(y.copy())
    report_samples = int(samples_per_cycle)
    phase_times = np.mod(np.asarray(times), ref.T)
    states_array = np.asarray(states)
    order = np.argsort(phase_times)
    phase_times = phase_times[order]
    states_array = states_array[order]
    # Uniform phase reporting avoids derivative/jerk metrics changing merely
    # because an integer downsampling index lands at a different sub-step.
    tl = np.linspace(0.0, ref.T, report_samples, endpoint=False)
    Y = np.vstack([
        np.interp(tl, phase_times, states_array[:, j])
        for j in range(states_array.shape[1])
    ])
    t = tl
    r = ref.evaluate(t)
    rows = []
    for i, ti in enumerate(t):
        d = dyn.diagnostics(float(ti), Y[:, i])
        rows.append(d)
    D = pd.DataFrame(rows)
    q, qdot, delta, deltadot, k_eff, b_eff, thetaeq_eff, kp_eff = Y
    qdd = D["qddot"].to_numpy(float)
    L = mech.paw_length_m
    toe_ax = np.asarray(r["hock_ax"]) - L * (qdot**2 * np.cos(q) + qdd * np.sin(q))
    toe_ay = np.asarray(r["hock_ay"]) - L * (qdot**2 * np.sin(q) - qdd * np.cos(q))
    out = pd.DataFrame({
        "time_s": tl, "cycle_percent": tl / ref.T * 100.0, "phase": D["phase"],
        "q_paw_rad_sim": q, "qdot_rad_s_sim": qdot, "qdd_rad_s2_sim": qdd,
        "hock_angle_rad_sim": D["theta"], "hock_angle_velocity_rad_s_sim": D["theta_dot"],
        "hock_x_m": D["hock_x"], "hock_y_m": D["hock_y"],
        "toe_x_m_sim": D["toe_x"], "toe_y_m_sim": D["toe_y"],
        "toe_vx_m_s_sim": D["toe_vx"], "toe_vy_m_s_sim": D["toe_vy"],
        "toe_ax_m_s2_sim": toe_ax, "toe_ay_m_s2_sim": toe_ay,
        "paw_compression_m": np.maximum(delta, 0.0), "paw_compression_velocity_m_s": deltadot,
        "normal_force_N": D["normal_force"], "target_load_N": D["target_load"],
        "tau_hock_Nm": D["tau_hock"], "tau_hock_raw_Nm": D["tau_hock_raw"],
        "tau_spring_Nm": D["tau_spring"], "tau_damping_Nm": D["tau_damping"],
        "torque_saturated": D["torque_saturated"],
        "k_hock_command_Nm_rad": D["cmd_k_hock"], "k_hock_effective_Nm_rad": k_eff,
        "b_hock_command_Nm_s_rad": D["cmd_b_hock"], "b_hock_effective_Nm_s_rad": b_eff,
        "theta_eq_command_rad": D["cmd_theta_eq"], "theta_eq_effective_rad": thetaeq_eff,
        "k_paw_command_N_m": D["cmd_k_paw"], "k_paw_effective_N_m": kp_eff,
        "q_paw_rad_ref": r["q_ref"], "hock_angle_rad_ref": r["theta_ref"],
        "toe_x_m_ref": r["toe_x"], "toe_y_m_ref": r["toe_y"],
        "toe_vx_m_s_ref": r["toe_vx"], "toe_vy_m_s_ref": r["toe_vy"],
        "toe_ax_m_s2_ref": r["toe_ax"], "toe_ay_m_s2_ref": r["toe_ay"],
    })
    theta_eq_rate = np.gradient(np.unwrap(thetaeq_eff), tl)
    spring_preload_torque = -k_eff * np.asarray(D["theta_error"], float)
    out["equilibrium_tuning_power_W"] = np.abs(spring_preload_torque * theta_eq_rate)
    out["joint_power_W"] = out["tau_hock_Nm"] * out["hock_angle_velocity_rad_s_sim"]
    return out.sort_values("cycle_percent").reset_index(drop=True)
