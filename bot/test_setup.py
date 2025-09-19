#!/usr/bin/env python3
"""
Test Setup Wrapper for RL Bot System
This is a convenience wrapper that runs the test setup from the setup directory.
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Run the test setup script from the setup directory."""
    setup_dir = Path(__file__).parent / "setup"
    test_setup_script = setup_dir / "test_setup.py"
    
    if not test_setup_script.exists():
        print(f"❌ Test setup script not found at: {test_setup_script}")
        sys.exit(1)
    
    # Run the actual test setup script
    try:
        subprocess.run([sys.executable, str(test_setup_script)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Test setup failed with exit code: {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n⚠️  Test setup interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()