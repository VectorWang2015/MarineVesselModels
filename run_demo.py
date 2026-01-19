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


def find_demo_files():
    """Find all Python demo files in the demos/ directory recursively."""
    demo_dir = os.path.join(os.path.dirname(__file__), 'demos')
    demo_files = []
    if os.path.exists(demo_dir):
        for root, dirs, files in os.walk(demo_dir):
            # Skip __pycache__ directories
            if '__pycache__' in root:
                continue
            for fname in files:
                if fname.endswith('.py') and fname != '__init__.py':
                    # Remove .py extension and optional demo_ prefix
                    demo_name = fname[:-3]
                    if demo_name.startswith('demo_'):
                        demo_name = demo_name[5:]
                    demo_files.append(demo_name)
    return sorted(demo_files)


def find_demo_path(demo_name):
    """Find the full path to a demo file given its name (with or without demo_ prefix)."""
    demo_dir = os.path.join(os.path.dirname(__file__), 'demos')
    if not os.path.exists(demo_dir):
        return None
    
    # Possible filename variations
    possible_filenames = [
        f"{demo_name}.py",
        f"demo_{demo_name}.py",
        f"{demo_name}"
    ]
    
    for root, dirs, files in os.walk(demo_dir):
        if '__pycache__' in root:
            continue
        for fname in files:
            if not fname.endswith('.py') or fname == '__init__.py':
                continue
            # Check if fname matches any possible filename
            for candidate in possible_filenames:
                if not candidate.endswith('.py'):
                    candidate += '.py'
                if fname == candidate:
                    return os.path.join(root, fname)
            # Also check if demo_name matches fname without .py and optional demo_ prefix
            base = fname[:-3]
            if base == demo_name or (base.startswith('demo_') and base[5:] == demo_name):
                return os.path.join(root, fname)
    
    return None


def find_demos_by_category():
    """Return a dict mapping category (subdirectory) to list of demo names."""
    demo_dir = os.path.join(os.path.dirname(__file__), 'demos')
    categories = {}
    if not os.path.exists(demo_dir):
        return categories
    
    for root, dirs, files in os.walk(demo_dir):
        if '__pycache__' in root:
            continue
        rel_root = os.path.relpath(root, demo_dir)
        if rel_root == '.':
            category = 'root'
        else:
            category = rel_root
        for fname in files:
            if not fname.endswith('.py') or fname == '__init__.py':
                continue
            # Extract demo name (without .py and optional demo_ prefix)
            demo_name = fname[:-3]
            if demo_name.startswith('demo_'):
                demo_name = demo_name[5:]
            categories.setdefault(category, []).append(demo_name)
    
    # Sort each category's demo list
    for cat in categories:
        categories[cat].sort()
    return categories


def list_demos(category=None):
    """List available demo files, optionally filtered by category."""
    if category is None:
        demos = find_demo_files()
        print("Available demos:")
        for demo in demos:
            print(f"  {demo}")
        return 0
    else:
        categories = find_demos_by_category()
        if category not in categories:
            print(f"Error: Category '{category}' not found")
            print("Available categories:")
            for cat in sorted(categories.keys()):
                print(f"  {cat}")
            return 1
        demos = categories[category]
        print(f"Demos in category '{category}':")
        for demo in demos:
            print(f"  {demo}")
        return 0


def list_categories():
    """List all available categories with demo counts."""
    categories = find_demos_by_category()
    print("Available categories:")
    for cat in sorted(categories.keys()):
        count = len(categories[cat])
        print(f"  {cat}: {count} demo{'s' if count != 1 else ''}")
    return 0


def run_demo(demo_name, extra_args):
    """Run a specific demo script."""
    demo_path = find_demo_path(demo_name)
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
  python run_demo.py --list --category identification
  python run_demo.py --list-categories
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
        '--list-categories',
        action='store_true',
        help='List all available categories'
    )
    parser.add_argument(
        '--category', '-c',
        help='Filter demos by category (use with --list)'
    )
    parser.add_argument(
        'args',
        nargs=argparse.REMAINDER,
        help='Additional arguments to pass to the demo script'
    )

    args = parser.parse_args()
    if args.list_categories:
        return list_categories()
    if args.list:
        return list_demos(category=args.category)
    elif args.demo:
        return run_demo(args.demo, args.args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
