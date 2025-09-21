#!/usr/bin/env python3
"""
Simple wrapper script that calls the main setup script.
This maintains backward compatibility while organizing scripts in the setup/ folder.
"""
import subprocess
import sys
from pathlib import Path
import argparse

def run_setup():
    """Run the main setup script."""
    bot_dir = Path(__file__).parent
    setup_script = bot_dir / "setup" / "setup.py"
    
    if not setup_script.exists():
        print("❌ Setup script not found at setup/setup.py")
        sys.exit(1)
    
    # Change to bot directory and run setup script
    cmd = [sys.executable, str(setup_script)] + sys.argv[1:]
    
    try:
        result = subprocess.run(cmd, cwd=bot_dir)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running setup: {e}")
        sys.exit(1)

def run_setup_gpu():
    """Run the GPU setup script from the setup directory."""
    setup_dir = Path(__file__).parent / "setup"
    gpu_setup_script = setup_dir / "setup_gpu.py"
    
    if not gpu_setup_script.exists():
        print(f"❌ GPU setup script not found at: {gpu_setup_script}")
        sys.exit(1)
    
    # Run the actual GPU setup script
    try:
        subprocess.run([sys.executable, str(gpu_setup_script)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ GPU setup failed with exit code: {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n⚠️  GPU setup interrupted by user")
        sys.exit(1)

def run_diagnose_gpu():
    """Run the diagnose GPU script from the setup directory."""
    setup_dir = Path(__file__).parent / "setup"
    diagnose_gpu_script = setup_dir / "setup_gpu.py"
    
    if not diagnose_gpu_script.exists():
        print(f"❌ Diagnose GPU script not found at: {diagnose_gpu_script}")
        sys.exit(1)
    
    # Run the actual GPU setup script
    try:
        subprocess.run([sys.executable, str(diagnose_gpu_script)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Diagnose GPU failed with exit code: {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n⚠️  Diagnose GPU interrupted by user")
        sys.exit(1)

def run_validate():
    """Run the test setup script from the setup directory."""
    setup_dir = Path(__file__).parent / "setup"
    test_setup_script = setup_dir / "validate_setup.py"
    
    if not test_setup_script.exists():
        print(f"❌ Validate setup script not found at: {test_setup_script}")
        sys.exit(1)
    
    # Run the actual test setup script
    try:
        subprocess.run([sys.executable, str(test_setup_script)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Validate setup failed with exit code: {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n⚠️  Validate setup interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("gpu", action="store_true")
    argparse.add_argument("validate", action="store_true")
    argparse.add_argument("diagnose_gpu", action="store_true")
    args = argparse.parse_args()

    if args.gpu:
        run_setup_gpu()
    elif args.validate:
        run_validate()
    elif args.diagnose_gpu:
        run_diagnose_gpu()
    else:
        run_setup()