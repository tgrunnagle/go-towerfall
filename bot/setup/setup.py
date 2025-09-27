"""
Main setup script for the RL Bot System.
Handles environment setup and basic dependency installation.
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    major, minor = sys.version_info.major, sys.version_info.minor
    
    if (major, minor) < (3, 12):
        print("Error: Python 3.12 is required for optimal CUDA support and compatibility.")
        print("Please install Python 3.12 and try again.")
        sys.exit(1)
    
    print(f"✓ Python {major}.{minor} detected")
    
    # Check for CUDA compatibility
    if (major, minor) >= (3, 13):
        print("⚠️  Warning: Python 3.13+ may have limited CUDA support.")
        print("   Python 3.12 is recommended for best compatibility.")
    elif (major, minor) == (3, 12):
        print("✓ Python 3.12 - Excellent CUDA compatibility")
    
    return (major, minor)


def check_uv_required():
    """Check if uv is available and exit if not."""
    try:
        result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ uv available: {result.stdout.strip()}")
            return True
        else:
            print("❌ uv is not working properly")
            print_uv_installation_instructions()
            sys.exit(1)
    except FileNotFoundError:
        print("❌ uv is required but not found")
        print_uv_installation_instructions()
        sys.exit(1)


def print_uv_installation_instructions():
    """Print instructions for installing uv."""
    print("\n💡 uv is required for this project. Please install it:")
    print("   Windows (PowerShell):")
    print("     powershell -ExecutionPolicy ByPass -c \"irm https://astral.sh/uv/install.ps1 | iex\"")
    print("   macOS/Linux:")
    print("     curl -LsSf https://astral.sh/uv/install.sh | sh")
    print("   Alternative (with pip):")
    print("     pip install uv")
    print("\n   After installation, restart your terminal and try again.")
    print("   Documentation: https://docs.astral.sh/uv/")


def create_virtual_environment(venv_path: Path, python_version: str = "3.12"):
    """Create a virtual environment using uv."""
    if venv_path.exists():
        print(f"✓ Virtual environment already exists at {venv_path}")
        return
    
    print(f"Creating virtual environment at {venv_path} with Python {python_version}...")
    
    try:
        cmd = ["uv", "venv", str(venv_path), "--python", python_version]
        subprocess.run(cmd, check=True)
        print(f"✓ Virtual environment created with uv at {venv_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        print(f"Make sure Python {python_version} is installed and available.")
        print("You can install Python 3.12 with:")
        print("  uv python install 3.12")
        sys.exit(1)


def install_dependencies(venv_path: Path, requirements_file: Path):
    """Install dependencies in the virtual environment using uv pip."""
    print("Installing dependencies with uv pip...")
    
    try:
        # Use uv pip with the virtual environment
        subprocess.run([
            "uv", "pip", "install", "-r", str(requirements_file)
        ], check=True, env={**os.environ, "VIRTUAL_ENV": str(venv_path)})
        print("✓ Dependencies installed successfully with uv pip")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("This might be due to:")
        print("  1. Network connectivity issues")
        print("  2. Invalid package versions in requirements.txt")
        print("  3. Python version compatibility issues")
        sys.exit(1)


def create_directories():
    """Create necessary directories for the RL bot system."""
    directories = [
        "data/models",
        "data/replays", 
        "data/metrics",
        "logs",
        "config"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def setup_git_ignore():
    """Update .gitignore to exclude generated files."""
    gitignore_path = Path("../.gitignore")  # Go up one level from bot/ to project root
    
    # Lines to add to .gitignore
    ignore_lines = [
        "",
        "# RL Bot System",
        "data/models/*.pth",
        "data/models/*.pkl",
        "data/replays/*.h5",
        "data/metrics/*.json",
        "bot/logs/*.log",
        "bot/logs/tensorboard/",
        "bot/.venv/",
        "__pycache__/",
        "*.pyc",
        "*.pyo",
        "*.egg-info/",
        ".pytest_cache/",
        ".coverage",
        "wandb/"
    ]
    
    if gitignore_path.exists():
        with open(gitignore_path, 'r') as f:
            existing_content = f.read()
        
        # Check if our section already exists
        if "# RL Bot System" in existing_content:
            print("✓ .gitignore already contains RL Bot System entries")
            return
    else:
        existing_content = ""
    
    # Append our ignore rules
    with open(gitignore_path, 'w') as f:
        f.write(existing_content)
        f.write('\n'.join(ignore_lines))
    
    print("✓ Updated .gitignore with RL Bot System entries")


def verify_installation(venv_path: Path):
    """Verify that key packages are installed correctly using uv pip."""
    key_packages = [
        "torch",
        "stable-baselines3", 
        "gymnasium",
        "numpy",
        "pyyaml"
    ]
    
    print("Verifying installation...")
    
    for package in key_packages:
        try:
            result = subprocess.run([
                "uv", "pip", "show", package
            ], capture_output=True, text=True, check=True, 
            env={**os.environ, "VIRTUAL_ENV": str(venv_path)})
            
            if result.returncode == 0:
                print(f"✓ {package} installed")
            else:
                print(f"✗ {package} not found")
        except subprocess.CalledProcessError:
            print(f"✗ {package} not found")


def main():
    parser = argparse.ArgumentParser(description="Setup RL Bot System environment (requires uv)")
    parser.add_argument("--venv-path", type=str, default=".venv", 
                       help="Path for virtual environment (default: .venv)")
    parser.add_argument("--skip-venv", action="store_true",
                       help="Skip virtual environment creation")
    parser.add_argument("--requirements", type=str, default="requirements.txt",
                       help="Requirements file path (default: requirements.txt)")
    parser.add_argument("--python", type=str, default="3.12",
                       help="Python version to use for virtual environment (default: 3.12)")
    
    args = parser.parse_args()
    
    print("Setting up RL Bot System...")
    print("=" * 50)
    
    # Check that uv is available (required)
    check_uv_required()
    
    # Check Python version
    python_version = check_python_version()
    
    # Create directories
    create_directories()
    
    # Setup .gitignore
    setup_git_ignore()
    
    if not args.skip_venv:
        # Create virtual environment
        venv_path = Path(args.venv_path)
        create_virtual_environment(venv_path, args.python)
        
        # Install dependencies
        requirements_file = Path(args.requirements)
        if not requirements_file.exists():
            print(f"❌ Requirements file not found: {requirements_file}")
            sys.exit(1)
        
        install_dependencies(venv_path, requirements_file)
        
        # Verify installation
        verify_installation(venv_path)
        
        print("\n" + "=" * 50)
        print("🎉 Setup completed successfully!")
        
        # Show activation instructions
        print(f"\nTo activate the virtual environment:")
        if os.name == 'nt':  # Windows
            print(f"  {venv_path}\\Scripts\\activate")
        else:  # Unix-like
            print(f"  source {venv_path}/bin/activate")
        
        # Show next steps
        print(f"\nNext steps:")
        print(f"  1. Activate the virtual environment (see above)")
        print(f"  2. For GPU support: python setup/gpu_setup.py")
        print(f"  3. Test installation: python setup/test_setup.py")
        print(f"  4. Run diagnostics: python setup/diagnose_gpu.py")
        
        # Show uv usage
        print(f"\nTo use uv with this environment:")
        print(f"  uv pip install <package>  # Install packages")
        print(f"  uv pip list               # List installed packages")
        print(f"  uv pip show <package>     # Show package info")
            
    else:
        print("\n" + "=" * 50)
        print("Setup completed successfully!")
        print("Note: Virtual environment creation was skipped.")
        print("Make sure to create and activate a virtual environment:")
        print(f"  uv venv --python {args.python}")
        print("  # Activate the environment, then:")
        print(f"  uv pip install -r {args.requirements}")


if __name__ == "__main__":
    main()