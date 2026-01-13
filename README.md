# MarineVesselModels

using:

```
https://github.com/VectorWang2015/USVWidgets
```

## Records

> Fossen_PRBS_nps_tt_xx.json

Fossen model, using PRBS control sequence for xx seconds, time step tt;  
nps indicats that experiment record generated through rps input PRBS instead of tau.  

data: dict; keys ["xs", "ys", "psis", "us", "vs", "rs", "dot_us", "dot_vs", "dot_rs", "Fxs", "Fys", "Nrs", "rps_l", "rps_r"]  

## Records (obsolete)

> Fossen_zigzag_xx_yy_zz_tt.npy

Fossen model, zigzag case with target psi xx degrees; base thrust xx N for each thrust; delta thrust for turing zz N; time step tt s.  

data: 9xn; [x, y, psi, u, v, r, tau1, tau2, tau3]  

> Fossen_nps_zigzag_xx_yy_zz_tt.npy

Fossen model with thrust model using nps as each thruts input, zigzag case with target psi xx degrees; base nps xx for each thrust; delta nps for turing zz N; time step tt s.  

data: 8xn; [x, y, psi, u, v, r, left_nps, right_nps]  

> KT_zigzag_xx_yy_tt.npy

First order non-linear response model, zigzag case with target psi xx degrees; turning rudder angle delta yy degrees; time step tt s.  

data: 7xn; [x, y, psi, u, v, r, delta]  

> Fossen_PRBS_tt_xx.npy
> partial_PRBS_tt_xx.npy

Fossen model, using PRBS control sequence for xx rounds, time step tt;  
Partial file for dot(state) data storage, which are exported from simulator directly.  

data: 9xn; [x, y, psi, u, v, r, tau1, tau2, tau3]  

## Running Demos

Use the wrapper script to run demos from the project root:

```bash
python run_demo.py --demo pid_heading
python run_demo.py --demo fossen_PRBS
python run_demo.py --list
```

The wrapper sets `PYTHONPATH` correctly so imports work without copying demo files to root.

## Simulators

The project provides several vessel simulator classes:

### VesselSimulator
Basic simulator implementing Fossen's marine vessel equations of motion.

### NoisyVesselSimulator
Extends `VesselSimulator` with Gaussian and Gauss-Markov noise models for control inputs and observations.

### SimplifiedEnvironmentalDisturbanceSimulator
Extends `VesselSimulator` with constant directional environmental forces (wind/wave).
- Takes environmental force magnitude and direction (global coordinates) as input
- Converts global force to body frame based on vessel heading
- Adds environmental force to control inputs
- Simplified model: force only, no yaw moment from environment

```python
from MarineVesselModels.simulator import SimplifiedEnvironmentalDisturbanceSimulator

simulator = SimplifiedEnvironmentalDisturbanceSimulator(
    hydro_params=sample_hydro_params_2,
    time_step=0.1,
    env_force_magnitude=10.0,  # Newtons
    env_force_direction=np.pi/4,  # radians (45°, northeast)
    model=Fossen,
)
```

## Development Notes

- Demo files use absolute imports (`from MarineVesselModels.simulator import...`)
- The wrapper script fixes import paths by setting `PYTHONPATH`
- Most demos create matplotlib plots and block on `plt.show()`
- GUI demos require PyQt6 (demo_EOT_main.py, simulatorGui.py)

