# MarineVesselModels Development Guide

## Running Demos

Use the wrapper script to run demos from the project root:

```bash
python run_demo.py --demo pid_heading
python run_demo.py --demo fossen_PRBS
python run_demo.py --list
```

The wrapper sets `PYTHONPATH` correctly so imports work without copying demo files to root.

## Dependencies

Core dependencies:
- numpy
- matplotlib  
- scipy
- PyQt6 (optional, for GUI demos)

Install with:
```bash
pip install -r requirements.txt
```

Or for conda:
```bash
conda install numpy matplotlib scipy
```

## Project Structure

- `MarineVesselModels/` - Core simulation models (Fossen, KT)
- `control/` - Controllers (PID, LQR, LOS, VO)
- `identification/` - Parameter estimation methods (LS, PRBS)
- `demos/` - Demonstration scripts
- `exp_data/` - Experimental data files
- `USVWidgets/` - PyQt6 GUI components

## Development Notes

- Demo files use absolute imports (`from MarineVesselModels.simulator import...`)
- The wrapper script fixes import paths by setting `PYTHONPATH`
- Most demos create matplotlib plots and block on `plt.show()`
- GUI demos require PyQt6 (demo_EOT_main.py, simulatorGui.py)

## Testing

No formal test suite currently. Demos serve as integration tests.

## Code Quality

No linting or type-checking configured. Consider adding:
- `ruff` for linting
- `black` for formatting  
- `mypy` for type checking