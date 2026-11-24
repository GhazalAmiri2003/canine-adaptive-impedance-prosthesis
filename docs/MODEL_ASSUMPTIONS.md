# Model assumptions and evidence boundary

## Mechanical scope

- Two-dimensional sagittal reduced-order model.
- Natural knee is preserved and the knee-to-hock motion is prescribed from the
  Dog G vision reference.
- Rigid shank and rigid paw links with an impedance-controlled hock joint.
- Paw compression is a separate vertical contact state; it does not translate
  the rigid hock point.
- The optimized controller changes equivalent hock stiffness `K`, damping `B`,
  equilibrium angle `theta_eq`, and equivalent paw stiffness `k_paw`.

## Calibrations

- Body mass: 30 kg.
- Torque saturation: 0.45 N m/kg.
- Effective COP lever: 22% of hock-to-toe length. This reduced-order value was
  selected so inverse dynamics of the natural reference remains near published
  Labrador hock moments while retaining the measured geometry.
- The GRF demand curve is constrained by the project literature targets for peak
  vertical force and vertical impulse.

## Scientific sources

- Ragetly et al. reported Labrador hock extensor moments around 0.28 to
  0.37 N m/kg in treadmill trot:
  https://doi.org/10.1111/j.1532-950X.2012.01042.x
- Shepherd and Rouse demonstrated a quasi-passive prosthetic ankle-foot design
  with continuously variable stiffness and a customizable torque-angle curve:
  https://doi.org/10.1109/TNSRE.2017.2750113
- Data-driven prosthesis research has modeled continuous stiffness, damping and
  equilibrium angle for impedance control:
  https://par.nsf.gov/servlets/purl/10385440

## Claims not made

- The model does not estimate true biological stiffness.
- The work metrics are not battery energy or metabolic cost.
- The loading is not synchronized Dog G force-plate data.
- The included TCN is not reported as trained because raw synchronized hind-limb
  6-axis IMU and FSR windows were not uploaded.
