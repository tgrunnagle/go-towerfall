#!/usr/bin/env python3
"""
GPU and CUDA diagnostic script for troubleshooting.
Provides detailed information about GPU setup and common issues.
"""
import subprocess
import sys
import platform
from pathlib import Path


def run_command(command, shell=True):
    """Run a command and return output, handling errors gracefully."""
    try:
        result = subprocess.run(
            command, 
            shell=shell, 
            capture_output=True, 
            text=True, 
            timeout=30
        )
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)


def check_nvidia_driver():
    """Check NVIDIA driver installation."""
    print("🔍 Checking NVIDIA Driver...")
    print("-" * 40)
    
    success, stdout, stderr = run_command("nvidia-smi")
    
    if success:
        print("✅ NVIDIA driver is installed and working")
        print("Driver information:")
        # Extract key info from nvidia-smi output
        lines = stdout.split('\n')
        for line in lines:
            if 'Driver Version' in line:
                print(f"  {line.strip()}")
            elif 'CUDA Version' in line:
                print(f"  {line.strip()}")
        
        # Show GPU details
        print("\nGPU Details:")
        for line in lines:
            if 'GeForce' in line or 'RTX' in line or 'GTX' in line or 'Quadro' in line:
                parts = line.split()
                if len(parts) > 1:
                    gpu_name = ' '.join([p for p in parts if any(x in p for x in ['GeForce', 'RTX', 'GTX', 'Quadro'])])
                    if gpu_name:
                        print(f"  GPU: {gpu_name}")
        return True
    else:
        print("❌ NVIDIA driver not found or not working")
        print(f"Error: {stderr}")
        print("\n💡 Solutions:")
        print("  1. Install NVIDIA drivers from: https://www.nvidia.com/drivers")
        print("  2. Restart your computer after installation")
        print("  3. Make sure your GPU is properly connected")
        return False


def check_cuda_toolkit():
    """Check CUDA toolkit installation."""
    print("\n🔍 Checking CUDA Toolkit...")
    print("-" * 40)
    
    # Check nvcc (CUDA compiler)
    success, stdout, stderr = run_command("nvcc --version")
    
    if success:
        print("✅ CUDA toolkit is installed")
        print("CUDA toolkit information:")
        lines = stdout.split('\n')
        for line in lines:
            if 'release' in line.lower():
                print(f"  {line.strip()}")
        return True
    else:
        print("❌ CUDA toolkit not found")
        print(f"Error: {stderr}")
        print("\n💡 Solutions:")
        print("  1. Download CUDA toolkit from: https://developer.nvidia.com/cuda-downloads")
        print("  2. Install CUDA 11.8 or 12.1 (recommended for PyTorch)")
        print("  3. Add CUDA to your PATH environment variable")
        print("  4. Restart your terminal/IDE after installation")
        return False


def check_pytorch_cuda():
    """Check PyTorch CUDA support."""
    print("\n🔍 Checking PyTorch CUDA Support...")
    print("-" * 40)
    
    try:
        import torch
        print(f"✅ PyTorch installed: {torch.__version__}")
        
        # Check if PyTorch was compiled with CUDA
        if torch.cuda.is_available():
            print("✅ PyTorch CUDA support: Available")
            print(f"  CUDA version in PyTorch: {torch.version.cuda}")
            print(f"  cuDNN version: {torch.backends.cudnn.version()}")
            
            # Check GPU devices
            gpu_count = torch.cuda.device_count()
            print(f"  GPU devices detected: {gpu_count}")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_props = torch.cuda.get_device_properties(i)
                gpu_memory = gpu_props.total_memory / (1024**3)
                print(f"    GPU {i}: {gpu_name}")
                print(f"      Memory: {gpu_memory:.1f} GB")
                print(f"      Compute Capability: {gpu_props.major}.{gpu_props.minor}")
            
            return True
        else:
            print("❌ PyTorch CUDA support: Not available")
            print(f"  PyTorch version: {torch.__version__}")
            
            # Check if this is a CPU-only build
            if '+cpu' in torch.__version__:
                print("  Issue: PyTorch was installed with CPU-only support")
                print("\n💡 Solution:")
                print("  Reinstall PyTorch with CUDA support:")
                print("  uv pip uninstall torch torchvision torchaudio")
                print("  uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            else:
                print("  Issue: CUDA runtime not found or incompatible")
                print("\n💡 Solutions:")
                print("  1. Install CUDA toolkit")
                print("  2. Check CUDA version compatibility with PyTorch")
                print("  3. Reinstall PyTorch with matching CUDA version")
            
            return False
            
    except ImportError:
        print("❌ PyTorch not installed")
        print("\n💡 Solution:")
        print("  Install PyTorch with CUDA support:")
        print("  uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        return False
    except Exception as e:
        print(f"❌ Error checking PyTorch: {e}")
        return False


def check_environment_variables():
    """Check relevant environment variables."""
    print("\n🔍 Checking Environment Variables...")
    print("-" * 40)
    
    import os
    
    cuda_vars = [
        'CUDA_PATH',
        'CUDA_HOME', 
        'CUDA_ROOT',
        'PATH'
    ]
    
    found_cuda_path = False
    
    for var in cuda_vars:
        value = os.environ.get(var)
        if value:
            if var == 'PATH':
                # Check if CUDA is in PATH
                cuda_in_path = any('cuda' in path.lower() for path in value.split(os.pathsep))
                if cuda_in_path:
                    print(f"✅ {var}: Contains CUDA paths")
                    found_cuda_path = True
                else:
                    print(f"⚠️  {var}: No CUDA paths found")
            else:
                print(f"✅ {var}: {value}")
                found_cuda_path = True
        else:
            if var != 'PATH':
                print(f"❌ {var}: Not set")
    
    if not found_cuda_path:
        print("\n💡 Solution:")
        print("  Add CUDA to your environment variables:")
        if platform.system() == "Windows":
            print("  1. Add C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1\\bin to PATH")
            print("  2. Set CUDA_PATH=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1")
        else:
            print("  1. Add /usr/local/cuda/bin to PATH")
            print("  2. Set CUDA_HOME=/usr/local/cuda")
    
    return found_cuda_path


def check_python_versions():
    """Check available Python versions."""
    print("\n🔍 Checking Python Versions...")
    print("-" * 40)
    
    current_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"Current Python: {current_version}")
    
    # Check for CUDA compatibility
    if sys.version_info >= (3, 13):
        print("⚠️  Python 3.13+ may have limited CUDA support")
        print("   Recommendation: Use Python 3.12 for best GPU acceleration")
    elif sys.version_info.major == 3 and sys.version_info.minor == 12:
        print("✅ Python 3.12 - Excellent CUDA compatibility")
    
    # Check if Python 3.12 is available
    python_312_cmds = ["python3.12", "python312", "py -3.12"]
    python_312_found = False
    
    for cmd in python_312_cmds:
        try:
            result = subprocess.run([cmd, "--version"], 
                                  capture_output=True, text=True, shell=True, timeout=5)
            if result.returncode == 0 and "Python 3.12" in result.stdout:
                print(f"✅ Python 3.12 available: {cmd}")
                python_312_found = True
                break
        except:
            continue
    
    if not python_312_found and sys.version_info.major != 3 or sys.version_info.minor != 12:
        print("❌ Python 3.12 not found")
        print("💡 Consider installing Python 3.12 for optimal CUDA support:")
        if platform.system() == "Windows":
            print("   - Download from: https://www.python.org/downloads/")
            print("   - Or use: winget install Python.Python.3.12")
        else:
            print("   - Ubuntu/Debian: sudo apt install python3.12")
            print("   - macOS: brew install python@3.12")
    
    return python_312_found


def check_system_info():
    """Display system information."""
    print("\n🔍 System Information...")
    print("-" * 40)
    
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python Version: {sys.version}")
    
    # Check if running in virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Running in virtual environment")
    else:
        print("⚠️  Not running in virtual environment")


def provide_installation_guide():
    """Provide step-by-step installation guide."""
    print("\n" + "="*60)
    print("📋 CUDA INSTALLATION GUIDE")
    print("="*60)
    
    system = platform.system()
    
    if system == "Windows":
        print("\n🪟 Windows Installation Steps:")
        print("0. Install Python 3.12 (Recommended):")
        print("   - Download from: https://www.python.org/downloads/release/python-3120/")
        print("   - Or use: winget install Python.Python.3.12")
        print("   - Make sure to check 'Add Python to PATH'")
        
        print("\n1. Install NVIDIA Drivers:")
        print("   - Go to https://www.nvidia.com/drivers")
        print("   - Download and install the latest driver for your GPU")
        print("   - Restart your computer")
        
        print("\n2. Create Virtual Environment with Python 3.12:")
        print("   - Run: python setup.py --python python3.12")
        print("   - Or: python setup.py --python \"py -3.12\"")
        
        print("\n3. Install CUDA Toolkit (Optional):")
        print("   - Go to https://developer.nvidia.com/cuda-downloads")
        print("   - Select Windows > x86_64 > your Windows version")
        print("   - Download CUDA 12.1 (recommended for PyTorch)")
        print("   - Run the installer with default settings")
        print("   - Restart your computer")
        
        print("\n4. Install PyTorch with CUDA:")
        print("   - Activate your virtual environment")
        print("   - Run: uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        
    elif system == "Linux":
        print("\n🐧 Linux Installation Steps:")
        print("1. Install NVIDIA Drivers:")
        print("   - Ubuntu/Debian: sudo apt install nvidia-driver-535")
        print("   - Or download from https://www.nvidia.com/drivers")
        print("   - Reboot: sudo reboot")
        
        print("\n2. Install CUDA Toolkit:")
        print("   - wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run")
        print("   - sudo sh cuda_12.1.0_530.30.02_linux.run")
        print("   - Follow the installer prompts")
        
        print("\n3. Set Environment Variables:")
        print("   - Add to ~/.bashrc:")
        print("     export PATH=/usr/local/cuda/bin:$PATH")
        print("     export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH")
        print("   - Run: source ~/.bashrc")
        
        print("\n4. Install PyTorch with CUDA:")
        print("   - uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    
    else:
        print(f"\n❓ {system} Installation:")
        print("Please refer to NVIDIA's official documentation for your operating system.")
    
    print("\n5. Test Installation:")
    print("   - Run: python bot/diagnose_gpu.py")
    print("   - Run: python bot/rl_bot_cli.py gpu-info")


def main():
    """Run complete GPU diagnostic."""
    print("🚀 GPU & CUDA Diagnostic Tool")
    print("="*60)
    
    # System info
    check_system_info()
    python_312_ok = check_python_versions()
    
    # Check components
    driver_ok = check_nvidia_driver()
    cuda_ok = check_cuda_toolkit()
    env_ok = check_environment_variables()
    pytorch_ok = check_pytorch_cuda()
    
    # Summary
    print("\n" + "="*60)
    print("📊 DIAGNOSTIC SUMMARY")
    print("="*60)
    
    components = [
        ("NVIDIA Driver", driver_ok),
        ("Python 3.12 Available", python_312_ok),
        ("CUDA Toolkit", cuda_ok),
        ("Environment Variables", env_ok),
        ("PyTorch CUDA", pytorch_ok)
    ]
    
    all_ok = True
    for name, status in components:
        icon = "✅" if status else "❌"
        print(f"{icon} {name}: {'OK' if status else 'NEEDS ATTENTION'}")
        if not status:
            all_ok = False
    
    if all_ok:
        print("\n🎉 All components are working correctly!")
        print("GPU acceleration should be available for training.")
    else:
        print("\n⚠️  Some components need attention.")
        provide_installation_guide()
    
    print(f"\n📝 For more help, see: bot/README_RL_BOT_SYSTEM.md")
    print("🔗 NVIDIA CUDA Installation: https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/")


if __name__ == "__main__":
    main()