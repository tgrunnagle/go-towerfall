#!/usr/bin/env python3
"""
Simple wrapper script that calls the main setup script.
This maintains backward compatibility while organizing scripts in the setup/ folder.
"""
import subprocess
import sys
from pathlib import Path

def main():
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

if __name__ == "__main__":
    main()