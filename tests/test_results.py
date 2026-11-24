from pathlib import Path
import json
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "results" / "csv"


def main():
    summary = pd.read_csv(CSV / "summary_metrics.csv")
    improvement = pd.read_csv(CSV / "improvement_summary.csv")
    heldout = pd.read_csv(CSV / "heldout_stride_paired_improvement.csv")
    checks = pd.read_csv(CSV / "physics_checks.csv")
    convergence = pd.read_csv(CSV / "numerical_convergence.csv")
    pars = json.loads((ROOT / "config" / "final_parameters.json").read_text())

    assert set(summary["condition"]) == {
        "static_single_device", "static_gait_specific", "adaptive_impedance"
    }
    assert (improvement["objective_improvement_vs_single_static_percent"] > 40.0).all()
    assert (improvement["angle_rmse_reduction_vs_single_static_percent"] > 65.0).all()
    assert (improvement["toe_rmse_reduction_vs_single_static_percent"] > 50.0).all()
    assert (heldout["adaptive_improvement_percent"] > 0.0).all()
    assert checks["pass"].all()
    assert np.isfinite(convergence.select_dtypes(include=[np.number])).all().all()
    assert pars["mechanical_parameters"]["torque_limit_Nm_per_kg"] == 0.45
    for gait in ("walk", "trot"):
        assert (ROOT / "results" / "videos" / f"DogG_{gait}_natural_static_adaptive.mp4").stat().st_size > 100_000
    print("All final-result checks passed.")


if __name__ == "__main__":
    main()
