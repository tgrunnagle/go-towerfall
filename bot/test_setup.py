#!/usr/bin/env python3
"""
Simple wrapper script that calls the test setup script.
This maintains backward compatibility while organizing scripts in the setup/ folder.
"""
import subprocess
import sys
from pathlib import Path

def main():
    """Run the test setup script."""
    bot_dir = Path(__file__).parent
    test_script = bot_dir / "setup" / "test_setup.py"
    
    if not test_script.exists():
        print("❌ Test script not found at setup/test_setup.py")
        sys.exit(1)
    
    # Change to bot directory and run test script
    cmd = [sys.executable, str(test_script)] + sys.argv[1:]
    
    try:
        result = subprocess.run(cmd, cwd=bot_dir)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running test: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()