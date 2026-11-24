# Final Results — Variable-Impedance Canine Prosthesis v3.0

## Scope

Subject-specific 2D sagittal simulation for Dog G. The natural knee is preserved. The model controls the distal hock/paw using phase-continuous equivalent impedance.

## Final comparison

### Walk

- Objective improvement vs one fixed device: **54.4%**
- Objective improvement vs gait-specific optimized static: **51.1%**
- Hock-angle RMSE reduction vs one fixed device: **87.1%**
- Toe-position RMSE reduction vs one fixed device: **68.6%**

### Trot

- Objective improvement vs one fixed device: **52.3%**
- Objective improvement vs gait-specific optimized static: **51.1%**
- Hock-angle RMSE reduction vs one fixed device: **71.7%**
- Toe-position RMSE reduction vs one fixed device: **57.9%**

## Held-out individual strides

- Trot: mean 41.7%, median 43.3%, worst stride 29.3% improvement.
- Walk: mean 39.0%, median 43.3%, worst stride 0.5% improvement.

## Interpretation

The visible difference comes from adding controlled equilibrium angle and damping to stiffness adaptation. This is a variable-impedance simulation, not a claim that a passive spring alone can reproduce the natural trajectory.

The adaptive profile is trained on the Dog G mean reference and tested on individual Dog G strides. Raw synchronized 6-axis IMU + FSR windows were not available, so the included TCN remains a deployment template. Existing phase-timing errors are used only for delay robustness.

## Fixed limitations

- Proximal knee-to-hock motion is prescribed from vision; whole-body dynamics are not solved.
- Vertical loading uses literature-calibrated targets rather than synchronized Dog G force-plate measurements.
- Work values are simulation proxies, not battery or metabolic energy.
- Equivalent/optimal stiffness is reported; no claim of true biological stiffness is made.