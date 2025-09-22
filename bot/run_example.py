#!/usr/bin/env python3
"""
Utility script for running RL bot system examples.

This script sets up the proper Python path and runs example scripts
from anywhere in the RL bot system.

Usage:
    python run_example.py <example_script_path>

Examples:
    python run_example.py rl_bot_system/evaluation/examples/simple_evaluation_demo.py
    python run_example.py rl_bot_system/training/examples/example_training_engine_usage.py
    python run_example.py rl_bot_system/models/examples/example_dqn_usage.py
"""

import sys
import os
import subprocess
from pathlib import Path


def main():
    """Run an example script with proper Python path setup."""
    if len(sys.argv) != 2:
        print("Usage: python run_example.py <example_script_path>")
        print("\nAvailable examples:")
        print("  rl_bot_system/evaluation/examples/simple_evaluation_demo.py")
        print("  rl_bot_system/training/examples/example_training_engine_usage.py")
        print("  rl_bot_system/training/examples/example_model_manager_usage.py")
        print("  rl_bot_system/training/examples/example_cohort_usage.py")
        sys.exit(1)
    
    example_path = sys.argv[1]
    
    # Get the bot directory (where this script is located)
    bot_dir = Path(__file__).parent.absolute()
    
    # Get the parent directory (project root) to add 'bot' to path
    project_root = bot_dir.parent
    
    # Construct full path to example script
    full_example_path = bot_dir / example_path
    
    if not full_example_path.exists():
        print(f"Error: Example script not found: {full_example_path}")
        sys.exit(1)
    
    # Set up environment
    env = os.environ.copy()
    
    # Add project root to Python path so 'bot' module can be imported
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"{project_root}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = str(project_root)
    
    # Run the example script
    try:
        print(f"Running example: {example_path}")
        print(f"Working directory: {bot_dir}")
        print("-" * 50)
        
        result = subprocess.run(
            [sys.executable, str(full_example_path)],
            cwd=bot_dir,
            env=env,
            check=True
        )
        
        print("-" * 50)
        print(f"Example completed successfully (exit code: {result.returncode})")
        
    except subprocess.CalledProcessError as e:
        print(f"Error running example: {e}")
        print(f"Exit code: {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nExample interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()