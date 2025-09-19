#!/usr/bin/env python3
"""
GPU Setup Wrapper for RL Bot System
This is a convenience wrapper that runs the GPU setup from the setup directory.
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Run the GPU setup script from the setup directory."""
    setup_dir = Path(__file__).parent / "setup"
    gpu_setup_script = setup_dir / "gpu_setup.py"
    
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

if __name__ == "__main__":
    main()