#!/usr/bin/env python3
"""
Wrapper script to run demo files from the demos/ directory.
Sets PYTHONPATH to include project root so imports work correctly.

Usage:
    python run_demo.py --demo demo_name
    python run_demo.py --list
"""

import sys
import os
import argparse
import subprocess
import importlib.util

def find_demo_files():
    """Find all Python demo files in the demos/ directory."""
    demo_dir = os.path.join(os.path.dirname(__file__), 'demos')
    demo_files = []
    if os.path.exists(demo_dir):
        for fname in os.listdir(demo_dir):
            if fname.endswith('.py') and fname != '__init__.py':
                demo_files.append(fname[:-3])  # Remove .py extension
    return sorted(demo_files)

def list_demos():
    """List all available demo files."""
    demos = find_demo_files()
    print("Available demos:")
    for demo in demos:
        print(f"  {demo}")
    return 0

def run_demo(demo_name, extra_args):
    """Run a specific demo script."""
    demo_dir = os.path.join(os.path.dirname(__file__), 'demos')
    demo_path = None
    
    # Try different possible file names
    possible_names = [
        f"{demo_name}.py",
        f"demo_{demo_name}.py",
        f"{demo_name}"
    ]
    
    for name in possible_names:
        if not name.endswith('.py'):
            name += '.py'
        path = os.path.join(demo_dir, name)
        if os.path.exists(path):
            demo_path = path
            break
    
    if not demo_path:
        # Check if demo_name is already a full filename in demos
        for fname in os.listdir(demo_dir):
            if fname.endswith('.py') and fname[:-3] == demo_name:
                demo_path = os.path.join(demo_dir, fname)
                break
    
    if not demo_path:
        print(f"Error: Demo '{demo_name}' not found in demos/ directory")
        print("Use --list to see available demos")
        return 1
    
    # Set PYTHONPATH to include project root
    project_root = os.path.dirname(__file__)
    env = os.environ.copy()
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = project_root + os.pathsep + env['PYTHONPATH']
    else:
        env['PYTHONPATH'] = project_root
    
    # Run the demo script
    cmd = [sys.executable, demo_path] + extra_args
    print(f"Running: {demo_path}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, env=env, cwd=project_root)
        return result.returncode
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
        return 130
    except Exception as e:
        print(f"Error running demo: {e}")
        return 1

def main():
    parser = argparse.ArgumentParser(
        description="Run demo scripts from the MarineVesselModels project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python run_demo.py --demo pid_heading
  python run_demo.py --demo demo_pid_heading
  python run_demo.py --demo fossen_PRBS
  python run_demo.py --list
"""
    )
    parser.add_argument(
        '--demo', '-d',
        help='Name of demo to run (with or without .py extension, with or without demo_ prefix)'
    )
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all available demos'
    )
    parser.add_argument(
        'args',
        nargs=argparse.REMAINDER,
        help='Additional arguments to pass to the demo script'
    )
    
    args = parser.parse_args()
    
    if args.list:
        return list_demos()
    elif args.demo:
        return run_demo(args.demo, args.args)
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())