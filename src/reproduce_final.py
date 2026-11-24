from __future__ import annotations

import argparse
import json
import subprocess

import run_pipeline as rp
from model import MechanicalParameters
from policies import StaticImpedancePolicy, build_adaptive_policy


def main():
    parser = argparse.ArgumentParser(description="Reproduce all final results from saved parameters")
    parser.add_argument("--skip-video", action="store_true")
    args = parser.parse_args()

    saved = json.loads((rp.ROOT / "config" / "final_parameters.json").read_text())
    mech = MechanicalParameters(**saved["mechanical_parameters"])
    refs = rp.load_mean_refs(mech)
    rp.refs_global = refs
    common = StaticImpedancePolicy(**saved["static_single_device"])
    gait_static = {
        gait: StaticImpedancePolicy(**saved["static_gait_specific"][gait])
        for gait in ("walk", "trot")
    }
    adaptive_pars = saved["adaptive_parameters"]
    adaptive = {
        gait: build_adaptive_policy(refs[gait], mech, adaptive_pars[gait])
        for gait in ("walk", "trot")
    }

    sims, summary, improvements = rp.evaluate_final(refs, mech, common, gait_static, adaptive)
    validation_raw, validation = rp.heldout_validation(mech, refs, common, adaptive)
    delay, sensitivity, convergence = rp.robustness_and_sensitivity(refs, mech, common, adaptive_pars)
    rp.make_plots(sims, summary, improvements, validation_raw, delay)
    rp.write_checks_and_docs(mech, summary, improvements, validation, delay, convergence, saved)

    if not args.skip_video:
        for gait in ("walk", "trot"):
            rp.make_video(
                gait, refs[gait], sims[(gait, "static_single_device")],
                sims[(gait, "adaptive_impedance")],
                rp.VIDEOS / f"DogG_{gait}_natural_static_adaptive.mp4",
            )
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(rp.VIDEOS / "DogG_walk_natural_static_adaptive.mp4"),
            "-i", str(rp.VIDEOS / "DogG_trot_natural_static_adaptive.mp4"),
            "-filter_complex", "hstack=inputs=2", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-crf", "21",
            str(rp.VIDEOS / "DogG_walk_trot_natural_static_adaptive.mp4"),
        ], check=True)

    print(improvements.to_string(index=False))


if __name__ == "__main__":
    main()
