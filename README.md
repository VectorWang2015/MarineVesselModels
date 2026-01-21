# MarineVesselModels

A personal project for marine vessel modeling, simulation, control, and system identification. This project provides tools for developing, testing, and evaluating control algorithms for unmanned surface vehicles (USVs) and other marine vessels.

## Features

- **Vessel Dynamics Simulation**: Fossen and KT models with Runge-Kutta integration
- **System Identification**: Multiple least-squares methods for parameter estimation
- **Guidance Algorithms**: LOS (Line-of-Sight) variants including adaptive and enhanced adaptive LOS
- **Control Systems**: PID, LQR, and specialized controllers for marine vessels
- **Environmental Modeling**: Noise models and environmental disturbance simulation
- **Interactive Demos**: 34 categorized demos for algorithm demonstration and evaluation
- **GUI Components**: PyQt6-based widgets for visualization and control

## Installation

```bash
# Clone the repository
git clone https://github.com/VectorWang2015/MarineVesselModels.git
cd MarineVesselModels

# Install core dependencies
pip install -r requirements.txt

# Optional: Install PyQt6 for GUI demos
# pip install PyQt6
```

Core dependencies: `numpy`, `matplotlib`, `scipy`

## Project Structure

```
MarineVesselModels/
├── MarineVesselModels/     # Core vessel models and simulators
├── control/               # Guidance and control algorithms
├── identification/        # System identification methods
├── demos/                 # Example demonstrations (categorized)
│   ├── data_generation/   # Data generation scripts (8)
│   ├── identification/    # Model identification demos (5)
│   ├── alos/             # Adaptive LOS variants (6)
│   ├── guidance/         # LOS guidance algorithms (6)
│   ├── control/          # PID/LQR controllers (5)
│   ├── utilities/        # Analysis tools (2)
│   └── environment/      # Environmental studies (1)
├── USVWidgets/           # PyQt6 GUI components
└── exp_data/             # Experimental data files
```

## Core Modules

### MarineVesselModels
- **Fossen.py**: Fossen's marine vessel equations of motion
- **KT.py**: First-order non-linear response (K-T) model
- **simulator.py**: Vessel simulators with noise and environmental disturbances
  - `VesselSimulator`: Basic vessel dynamics simulator
  - `NoisyVesselSimulator`: Adds Gaussian and Gauss-Markov noise to controls/observations
  - `SimplifiedEnvironmentalDisturbanceSimulator`: Adds constant directional environmental forces
- **noises.py**: Gaussian and Gauss-Markov noise generators
- **thrusters.py**: Thruster models and conversions
- **stepper.py**: Runge-Kutta integration for time-invariant systems

### Control Module
- **los.py**: Line-of-Sight guidance algorithms
  - `LOSGuider`: Basic LOS guidance
  - `FixedDistLOSGuider`: LOS with fixed lookahead distance
  - `DynamicDistLOSGuider`: Adaptive lookahead distance based on cross-track error
  - `AdaptiveLOSGuider`: Adaptive LOS with integral action
  - `EnhancedAdaptiveLOSGuider`: Enhanced adaptive LOS with conditional integration and reset mechanisms
- **pid.py**: PID controllers
  - `PID`: Standard PID controller
  - `PIDAW`: PID with anti-windup and derivative filter
  - `DoubleLoopHeadingPID`: Double-loop heading PID controller
- **lqr.py**: Linear Quadratic Regulators
  - `HeadingLQR`: LQR for heading control
  - `VeloLQR`: LQR for velocity control
- **controller.py**: Specialized controllers
  - `DualThrustUSVStopAndAdjustController`: Stop-and-adjust controller for dual-thrust USVs
- **vo.py**: Velocity Obstacle algorithms
  - `HeadingControlVO`: Velocity obstacle for heading control
  - `ColregVO`: COLREGs-compliant velocity obstacle
- **l1.py**: L1 guidance algorithm (`L1Guider`)

### Identification Module
- **least_square_methods.py**: Core identification algorithms
  - `LeastSquareFirstOrderNonLinearResponse`: LS for first-order non-linear response models
  - `RecursiveLeastSquareFirstOrderNonLinearResponse`: RLS variant
  - `LeastSquareFossen`: LS for Fossen model parameters
  - `LeastSquareFossenSG`: LS with Savitzky-Golay filtering
  - `RecursiveLeastSquareFossen`: RLS for Fossen model
  - `AlternatingLeastSquareFossen`: Alternating LS for Fossen model
- **improved_ls.py**: Enhanced least-squares methods
  - `WeightedRidgeLSFossen`: Weighted ridge regression for Fossen model
  - `WeightedRidgeLSThruster`: Weighted ridge regression for thruster model
  - `WeightedRidgeLSFossenRps`: Weighted ridge regression for Fossen model with RPS inputs
  - `AltLS`: Alternating least squares
- **PRBS.py**: Pseudo-Random Binary Sequence generation and analysis

### USVWidgets (GUI Components)
- **control.py**: Engine order telegraph widget (`EngineOrderTele`)
- **canvas.py**: Canvas for vessel visualization
- **shiphsi.py**: Horizontal situation indicator (HSI)
- **shipvsi.py**: Vertical speed indicator
- **propeller.py**: Propeller visualization
- **rudder.py**: Rudder visualization
- **palette_dark.py**: Dark color palette for PyQt6
### Utility Scripts
- **plot_utils.py**: Visualization utilities for vessel pose plotting and coordinate transformations
- **fossen_main.py**: Example script demonstrating Fossen model usage
- **simulatorGui.py**: PyQt6-based simulator GUI application
- **xinput_support.py**: XInput gamepad support for interactive control

## Demos

The project includes 33 demonstration scripts organized into categories:

### Data Generation (8 demos)
- **PRBS Experiments**: `fossen_PRBS`, `fossen_PRBS_debug`, `noisy_fossen_PRBS`, `fossen_n_input_PRBS`
- **Zigzag Maneuvers**: `fossen_zigzag`, `kt_zigzag`, `noisy_fossen_zigzag`, `fossen_n_input_zigzag`

### Identification (5 demos)
- **Model Identification**: `fossen_identification`, `kt_identification`, `thruster_identification`
- **Data Analysis**: `fossen_data_analysis`, `fossen_nps_identification`

### ALOS - Adaptive LOS (6 demos)
- **Comparison Studies**: `alos_comparison`, `enhanced_alos_comparison`, `zigzag_los_comparison`
- **Parameter Tuning**: `alos_tuning`, `tune_enhanced_alos`
- **Ablation Study**: `alos_ablation_study`
- **Multi-segment Paths**: `alos_multisegment`

### Guidance (6 demos)
- **LOS Variants**: `los_double_loop`, `los_heading_only`, `los_lqr`
- **Environmental Effects**: `los_guiders_env_disturbance`
- **PID Integration**: `los_guiders_pid_double_loop`

### Control (5 demos)
- **PID Controllers**: `pid_heading`, `pid_velo`
- **LQR Controllers**: `lqr_heading`, `LQR_velo`
- **Specialized Controllers**: `stop_and_adjust_controller`

### Utilities (2 demos)
- **Data Validation**: `check_nps_exp_data`
- **Spectral Analysis**: `check_PSD`

### Environment (1 demo)
- **Disturbance Studies**: `simplified_env_disturbance`



## Running Demos

Use the `run_demo.py` wrapper script to execute demonstrations:

```bash
# List all available demos
python run_demo.py --list

# List demos by category
python run_demo.py --list-categories
python run_demo.py --list --category identification

# Run a specific demo (with or without 'demo_' prefix)
python run_demo.py --demo pid_heading
python run_demo.py --demo demo_pid_heading
python run_demo.py --demo fossen_PRBS

# Pass arguments to demo scripts
python run_demo.py --demo alos_comparison -- --test-mode
```

### Key Features of `run_demo.py`
- **Recursive Search**: Finds demos in all subdirectories
- **Category Support**: Filter and list by category
- **Backward Compatibility**: Works with old demo names (with/without `demo_` prefix)
- **PYTHONPATH Management**: Sets proper import paths automatically

## Example Usage

```python
from MarineVesselModels.simulator import VesselSimulator
from MarineVesselModels.Fossen import sample_hydro_params_2
from control.pid import DoubleLoopHeadingPID

# Create a vessel simulator
simulator = VesselSimulator(
    hydro_params=sample_hydro_params_2,
    time_step=0.1,
    init_state=np.array([0, 0, 0, 0, 0, 0]).reshape([6, 1])
)

# Create a PID controller
controller = DoubleLoopHeadingPID(
    Kp=1.0, Ki=0.1, Kd=0.5,
    Kp_inner=2.0, Ki_inner=0.2, Kd_inner=0.3,
    time_step=0.1
)

# Run a simple control loop
for _ in range(100):
    # Get current heading
    current_heading = simulator.state[2, 0]
    
    # Compute control (target heading = 30°)
    control = controller.step(current_heading, np.deg2rad(30))
    
    # Apply control to simulator
    simulator.step(np.array([control, 0, 0]))
```

## Data Files

Experimental data is stored in `exp_data/`:
- **PRBS Experiments**: `Fossen_PRBS_nps_*.json` - PRBS input experiments with RPS inputs
- **Zigzag Maneuvers**: `*_zigzag_*.npy` - Zigzag maneuver data for various models
- **Partial Data**: `partial_PRBS_*.npy` - Partial derivative data for system identification

File naming convention: `{model}_{experiment_type}_{parameters}_{time_step}.{format}`

## Development

### Adding New Demos
1. Place demo script in appropriate category directory under `demos/`
2. Use `demo_` prefix for filename (e.g., `demo_new_feature.py`)
3. Use absolute imports: `from MarineVesselModels.simulator import ...`
4. The script will be automatically discovered by `run_demo.py`

## Acknowledgments

- **Fossen's Marine Craft Dynamics**: Mathematical models based on Fossen's equations
- **USVWidgets**: Companion GUI library available at [github.com/VectorWang2015/USVWidgets](https://github.com/VectorWang2015/USVWidgets)
