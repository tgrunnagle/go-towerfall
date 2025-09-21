"""
GPU setup script for the RL Bot System.
This script should be run AFTER activating the virtual environment.
"""
import subprocess
import sys
import json
from pathlib import Path


def check_virtual_environment():
    """Check if we're running in a virtual environment."""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✓ Running in virtual environment")
        return True
    else:
        print("❌ Not running in virtual environment")
        print("Please activate the virtual environment first:")
        print("  # Windows:")
        print("  .venv\\Scripts\\activate")
        print("  # Unix/macOS:")
        print("  source .venv/bin/activate")
        return False


def check_torch_installed():
    """Check if PyTorch is already installed."""
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} already installed")
        return True, torch.__version__
    except ImportError:
        print("✗ PyTorch not found")
        return False, None


def check_gpu_availability():
    """Check for GPU availability and CUDA support."""
    print("Checking GPU availability...")
    
    try:
        import torch
        import warnings
        
        # Capture warnings to detect compatibility issues
        compatibility_warning = None
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cuda_available = torch.cuda.is_available()
            
            # Also trigger device properties check to catch compatibility warnings
            if cuda_available and torch.cuda.device_count() > 0:
                try:
                    _ = torch.cuda.get_device_properties(0)
                except:
                    pass
            
            # Check for GPU compatibility warnings
            for warning in w:
                if "not compatible with the current PyTorch installation" in str(warning.message):
                    compatibility_warning = str(warning.message)
                    break
        
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            cuda_version = torch.version.cuda
            print(f"✓ CUDA available: {cuda_version}")
            print(f"✓ GPU devices found: {gpu_count}")
            print(f"✓ Primary GPU: {gpu_name}")
            
            # Check GPU memory
            if gpu_count > 0:
                gpu_props = torch.cuda.get_device_properties(0)
                gpu_memory = gpu_props.total_memory / (1024**3)
                compute_capability = f"{gpu_props.major}.{gpu_props.minor}"
                
                print(f"✓ GPU memory: {gpu_memory:.1f} GB")
                print(f"✓ Compute capability: {compute_capability}")
                
                if gpu_memory < 4.0:
                    print("⚠ Warning: GPU has less than 4GB memory. Consider using smaller models or CPU training.")
                elif gpu_memory >= 8.0:
                    print("✓ GPU has sufficient memory for large models")
                
                # Check for compatibility issues (RTX 5090 has compute capability 12.0)
                if float(compute_capability) >= 12.0 or compatibility_warning:
                    print(f"\n⚠️  GPU Compatibility Warning:")
                    if compatibility_warning:
                        print(f"   {compatibility_warning}")
                    else:
                        print(f"   RTX 5090 with compute capability {compute_capability} may have limited PyTorch support")
                    
                    print(f"\n💡 Solutions for RTX 5090 / Compute Capability {compute_capability}:")
                    print(f"   1. Install PyTorch nightly build:")
                    print(f"      uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121")
                    print(f"   2. Or wait for official PyTorch release with sm_120 support")
                    print(f"   3. For now, training may work but with reduced performance")
                    return "compatible_with_warning"
            
            return True
        else:
            print("✗ CUDA not available")
            print("Possible reasons:")
            print("  1. No NVIDIA GPU installed")
            print("  2. NVIDIA drivers not installed")
            print("  3. CUDA toolkit not installed")
            print("  4. PyTorch installed without CUDA support")
            print("  5. GPU compute capability not supported by PyTorch")
            return False
            
    except ImportError:
        print("✗ PyTorch not installed - cannot check GPU availability")
        return False
    except Exception as e:
        print(f"✗ Error checking GPU: {e}")
        return False


def install_gpu_pytorch():
    """Install GPU-enabled PyTorch using uv pip."""
    print("Installing GPU-optimized PyTorch...")
    
    # Install CUDA-enabled PyTorch
    gpu_packages = ["torch", "torchvision", "torchaudio"]
    cuda_index = "https://download.pytorch.org/whl/cu121"
    
    try:
        subprocess.run([
            "uv", "pip", "install", "--upgrade",
            "--index-url", cuda_index,
            *gpu_packages
        ], check=True)
        print("✓ GPU-optimized PyTorch installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install GPU-optimized PyTorch: {e}")
        print("This might be due to:")
        print("  1. CUDA version compatibility issues")
        print("  2. Network connectivity to PyTorch index")
        print("  3. Python version compatibility")
        return False


def install_cpu_pytorch():
    """Install CPU-only PyTorch using uv pip."""
    print("Installing CPU-only PyTorch...")
    
    gpu_packages = ["torch", "torchvision", "torchaudio"]
    
    try:
        subprocess.run([
            "uv", "pip", "install", "--upgrade",
            *gpu_packages
        ], check=True)
        print("✓ CPU-only PyTorch installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install CPU-only PyTorch: {e}")
        return False


def main():
    print("RL Bot System - GPU Setup")
    print("=" * 40)
    
    # Check if we're in a virtual environment
    if not check_virtual_environment():
        sys.exit(1)
    
    # Check if PyTorch is already installed
    torch_installed, torch_version = check_torch_installed()
    
    if torch_installed:
        # Check if current PyTorch has GPU support
        gpu_available = check_gpu_availability()
        
        if gpu_available == True:
            print("\n🎉 GPU acceleration is already available!")
            print("   Training will automatically use GPU when possible.")
            print("   You can force CPU training by setting device='cpu' in config.")
            return
        elif gpu_available == "compatible_with_warning":
            print("\n⚠️  GPU acceleration is available but with compatibility warnings.")
            print("   Training should work but may have reduced performance.")
            
            choice = input("\nDo you want to install PyTorch nightly for better RTX 5090 support? (y/N): ").lower().strip()
            
            if choice in ['y', 'yes']:
                print("Installing PyTorch nightly build...")
                try:
                    subprocess.run([
                        "uv", "pip", "install", "--upgrade", "--pre",
                        "torch", "torchvision", "torchaudio",
                        "--index-url", "https://download.pytorch.org/whl/nightly/cu121"
                    ], check=True)
                    print("✓ PyTorch nightly installed")
                    
                    print("\n✓ Checking GPU availability with nightly build...")
                    gpu_available_after = check_gpu_availability()
                    
                    if gpu_available_after == True:
                        print("\n🎉 GPU acceleration is now fully compatible!")
                    else:
                        print("\n⚠️  Still some compatibility issues, but should work for training.")
                        
                except subprocess.CalledProcessError as e:
                    print(f"❌ Failed to install PyTorch nightly: {e}")
            else:
                print("\nKeeping current PyTorch installation.")
                print("Training should still work with your RTX 5090.")
            return
        else:
            print("\n⚠️  PyTorch is installed but without GPU support.")
            choice = input("Do you want to reinstall PyTorch with GPU support? (y/N): ").lower().strip()
            
            if choice in ['y', 'yes']:
                if install_gpu_pytorch():
                    print("\n✓ Checking GPU availability after installation...")
                    gpu_available = check_gpu_availability()
                    
                    if gpu_available:
                        print("\n🎉 GPU acceleration is now available!")
                    else:
                        print("\n⚠️  GPU support still not available. Check your CUDA installation.")
                else:
                    print("\n❌ Failed to install GPU-enabled PyTorch")
            else:
                print("\nKeeping current CPU-only PyTorch installation.")
    else:
        print("\nPyTorch not found. Choose installation type:")
        print("1. GPU-enabled PyTorch (recommended if you have NVIDIA GPU)")
        print("2. CPU-only PyTorch")
        
        while True:
            choice = input("Enter choice (1 or 2): ").strip()
            
            if choice == "1":
                if install_gpu_pytorch():
                    print("\n✓ Checking GPU availability...")
                    gpu_available = check_gpu_availability()
                    
                    if gpu_available == True:
                        print("\n🎉 GPU acceleration is available!")
                    elif gpu_available == "compatible_with_warning":
                        print("\n⚠️  GPU acceleration available but with compatibility warnings.")
                        print("   Consider installing PyTorch nightly for better RTX 5090 support.")
                    else:
                        print("\n⚠️  GPU support not available. Check your CUDA installation.")
                        print("You can still use CPU training.")
                break
            elif choice == "2":
                if install_cpu_pytorch():
                    print("\n✓ CPU-only PyTorch installed successfully")
                    print("Training will use CPU only.")
                break
            else:
                print("Please enter 1 or 2")
    
    print("\n" + "=" * 40)
    print("GPU setup completed!")
    print("\nNext steps:")
    print("  1. Test installation: python setup/test_setup.py")
    print("  2. Run diagnostics: python setup/diagnose_gpu.py")
    print("  3. Start training: python rl_bot_cli.py train start")


if __name__ == "__main__":
    main()