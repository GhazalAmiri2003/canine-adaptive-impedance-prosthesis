from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


def wrap_pi(x):
    return (np.asarray(x) + np.pi) % (2.0 * np.pi) - np.pi


def periodic_smooth(y: np.ndarray, harmonics: int = 6) -> np.ndarray:
    """Low-pass a 0..100% gait curve while preserving periodic closure."""
    y = np.asarray(y, float)
    core = y[:-1].copy() if y.size > 2 else y.copy()
    spec = np.fft.rfft(core)
    if harmonics + 1 < spec.size:
        spec[harmonics + 1 :] = 0.0
    out = np.fft.irfft(spec, n=core.size)
    return np.r_[out, out[0]]


class PeriodicSignal:
    def __init__(self, t: np.ndarray, y: np.ndarray, preserve_drift: bool = False):
        t = np.asarray(t, float)
        y = np.asarray(y, float)
        self.T = float(t[-1])
        if preserve_drift:
            self.slope = float((y[-1] - y[0]) / self.T)
            residual = y - (y[0] + self.slope * t)
            self.base = float(y[0])
        else:
            self.slope = 0.0
            residual = y - np.linspace(0.0, 1.0, y.size) * (y[-1] - y[0])
            self.base = float(residual[0])
            residual = residual - residual[0]
        residual[-1] = residual[0]
        self.spline = CubicSpline(t, residual, bc_type="periodic")

    def __call__(self, t, nu: int = 0):
        tl = np.mod(t, self.T)
        if nu == 0:
            return self.base + self.slope * tl + self.spline(tl)
        if nu == 1:
            return self.slope + self.spline(tl, 1)
        return self.spline(tl, nu)


@dataclass
class GaitReference:
    gait_type: str
    T: float
    stance_fraction: float
    shank_length_m: float
    paw_length_m: float
    knee_x: PeriodicSignal
    knee_y: PeriodicSignal
    hock_x: PeriodicSignal
    hock_y: PeriodicSignal
    toe_x: PeriodicSignal
    toe_y: PeriodicSignal
    q_ref: PeriodicSignal
    phi_prox: PeriodicSignal
    theta_ref: PeriodicSignal
    source_id: str = "mean"

    @classmethod
    def from_standard_csv(
        cls,
        path: str | Path,
        gait_type: str,
        shank_length_m: float,
        paw_length_m: float,
        harmonics: int = 6,
    ) -> "GaitReference":
        df = pd.read_csv(path)
        sub = df[df["gait_type"].str.lower().eq(gait_type.lower())].sort_values("cycle_percent")
        if sub.empty:
            raise ValueError(f"No {gait_type} rows in {path}")
        return cls._from_standard_frame(sub, gait_type, shank_length_m, paw_length_m, harmonics, "DogG_mean")

    @classmethod
    def _from_standard_frame(
        cls,
        sub: pd.DataFrame,
        gait_type: str,
        shank_length_m: float,
        paw_length_m: float,
        harmonics: int,
        source_id: str,
    ) -> "GaitReference":
        cp = sub["cycle_percent"].to_numpy(float)
        T = float(sub["mean_stride_duration_s"].iloc[0])
        stance_fraction = float(sub["mean_stance_percent"].iloc[0]) / 100.0
        t = cp / 100.0 * T

        def sig(column: str, scale: float = 1.0, drift: bool = False, angle: bool = False):
            y = sub[column].to_numpy(float) * scale
            if angle:
                y = np.unwrap(y)
            y = periodic_smooth(y, harmonics)
            return PeriodicSignal(t, y, preserve_drift=drift)

        return cls(
            gait_type=gait_type.lower(),
            T=T,
            stance_fraction=stance_fraction,
            shank_length_m=float(shank_length_m),
            paw_length_m=float(paw_length_m),
            knee_x=sig("knee_x_norm_shank", shank_length_m, True),
            knee_y=sig("knee_y_norm_shank", shank_length_m),
            hock_x=sig("hock_x_norm_shank", shank_length_m, True),
            hock_y=sig("hock_y_norm_shank", shank_length_m),
            toe_x=sig("toe_x_norm_shank", shank_length_m, True),
            toe_y=sig("toe_y_norm_shank", shank_length_m),
            q_ref=sig("paw_angle_rad_ref", 1.0, angle=True),
            phi_prox=sig("proximal_angle_rad_ref", 1.0, angle=True),
            theta_ref=sig("hock_relative_angle_rad_ref", 1.0, angle=True),
            source_id=source_id,
        )

    def phase_fraction(self, t):
        return np.mod(np.asarray(t, float), self.T) / self.T

    def phase(self, t):
        stance = self.phase_fraction(t) < self.stance_fraction
        if np.ndim(stance) == 0:
            return "stance" if bool(stance) else "swing"
        return np.where(stance, "stance", "swing")

    def evaluate(self, t):
        return {
            "knee_x": self.knee_x(t), "knee_y": self.knee_y(t),
            "hock_x": self.hock_x(t), "hock_y": self.hock_y(t),
            "hock_vx": self.hock_x(t, 1), "hock_vy": self.hock_y(t, 1),
            "hock_ax": self.hock_x(t, 2), "hock_ay": self.hock_y(t, 2),
            "toe_x": self.toe_x(t), "toe_y": self.toe_y(t),
            "toe_vx": self.toe_x(t, 1), "toe_vy": self.toe_y(t, 1),
            "toe_ax": self.toe_x(t, 2), "toe_ay": self.toe_y(t, 2),
            "q_ref": self.q_ref(t), "qdot_ref": self.q_ref(t, 1), "qdd_ref": self.q_ref(t, 2),
            "phi_prox": self.phi_prox(t), "phidot_prox": self.phi_prox(t, 1),
            "theta_ref": self.theta_ref(t), "thetadot_ref": self.theta_ref(t, 1),
        }


def build_subject_stride_references(
    targets_csv: str | Path,
    dog_id: str,
    shank_length_m: float,
    paw_length_m: float,
    harmonics: int = 5,
) -> list[GaitReference]:
    """Create held-out stride references without changing the original records."""
    raw = pd.read_csv(targets_csv)
    raw = raw[raw["dog_id"].astype(str).eq(str(dog_id))].copy()
    refs: list[GaitReference] = []
    for stride_id, g in raw.groupby("stride_id", sort=True):
        g = g.sort_values("cycle_percent")
        if g["cycle_percent"].nunique() < 80:
            continue
        L = float(g["shank_length_m"].median())
        frame = pd.DataFrame({
            "cycle_percent": g["cycle_percent"].to_numpy(float),
            "mean_stride_duration_s": float(g["stride_duration_s"].iloc[0]),
            "mean_stance_percent": float(g["stance_percent"].iloc[0]),
            "knee_x_norm_shank": 0.0,
            "knee_y_norm_shank": 0.0,
            "hock_x_norm_shank": g["hock_x_rel_knee_m"].to_numpy(float) / L,
            "hock_y_norm_shank": g["hock_y_rel_knee_m"].to_numpy(float) / L,
            "toe_x_norm_shank": g["toe_x_rel_knee_m"].to_numpy(float) / L,
            "toe_y_norm_shank": g["toe_y_rel_knee_m"].to_numpy(float) / L,
        })
        q = np.unwrap(np.arctan2(
            frame["toe_y_norm_shank"] - frame["hock_y_norm_shank"],
            frame["toe_x_norm_shank"] - frame["hock_x_norm_shank"],
        ))
        prox = np.unwrap(np.arctan2(-frame["hock_y_norm_shank"], -frame["hock_x_norm_shank"]))
        theta = np.unwrap(wrap_pi(q - prox))
        while np.median(theta) > 0:
            theta -= 2.0 * np.pi
        frame["paw_angle_rad_ref"] = q
        frame["proximal_angle_rad_ref"] = prox
        frame["hock_relative_angle_rad_ref"] = theta
        gait = str(g["gait_type"].iloc[0]).lower()
        refs.append(GaitReference._from_standard_frame(
            frame, gait, shank_length_m, paw_length_m, harmonics, str(stride_id)
        ))
    return refs
