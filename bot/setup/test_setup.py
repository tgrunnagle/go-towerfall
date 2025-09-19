#!/usr/bin/env python3
"""
Test script to verify the RL bot system setup.
"""
import sys
import traceback
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        # Import setup package to trigger sys.path modification
        import setup
        
        # Ensure bot directory is in path for imports
        import sys
        bot_dir = str(Path(__file__).parent.parent)
        if bot_dir not in sys.path:
            sys.path.insert(0, bot_dir)
        
        # Test configuration system
        from rl_bot_system.config.config_manager import config_manager
        from rl_bot_system.config.base_config import RLBotSystemConfig
        print("✓ Configuration system imports successful")
        
        # Test that we can load default config
        config = config_manager.get_default_config()
        print("✓ Default configuration loaded successfully")
        
        # Test configuration validation
        config.validate()
        print("✓ Configuration validation successful")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        traceback.print_exc()
        return False


def test_directories():
    """Test that all required directories exist."""
    print("\nTesting directories...")
    
    required_dirs = [
        "rl_bot_system",
        "rl_bot_system/config",
        "rl_bot_system/models",
        "rl_bot_system/training",
        "rl_bot_system/evaluation",
        "rl_bot_system/environment",
        "rl_bot_system/rules_based",
        "rl_bot_system/storage",
        "rl_bot_system/server",
        "rl_bot_system/utils",
        "data",
        "data/models",
        "data/replays",
        "data/metrics",
        "logs",
        "config"
    ]
    
    all_exist = True
    for directory in required_dirs:
        path = Path(directory)
        if path.exists():
            print(f"✓ {directory}")
        else:
            print(f"✗ {directory} - missing")
            all_exist = False
    
    return all_exist


def test_config_files():
    """Test that configuration files exist and are valid."""
    print("\nTesting configuration files...")
    
    config_files = [
        "config/default.yaml",
        "config/quick_training.yaml",
        "config/gpu_training.yaml"
    ]
    
    all_valid = True
    for config_file in config_files:
        path = Path(config_file)
        if path.exists():
            try:
                from rl_bot_system.config.base_config import RLBotSystemConfig
                config = RLBotSystemConfig.from_yaml(path)
                config.validate()
                print(f"✓ {config_file} - valid")
            except Exception as e:
                print(f"✗ {config_file} - invalid: {e}")
                all_valid = False
        else:
            print(f"✗ {config_file} - missing")
            all_valid = False
    
    return all_valid


def test_gpu_availability():
    """Test GPU availability and CUDA support."""
    print("\nTesting GPU availability...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            cuda_version = torch.version.cuda
            print(f"✓ CUDA available: {cuda_version}")
            print(f"✓ GPU devices: {gpu_count}")
            print(f"✓ Primary GPU: {gpu_name}")
            
            if gpu_count > 0:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"✓ GPU memory: {gpu_memory:.1f} GB")
            
            return True
        else:
            print("ℹ CUDA not available - CPU training only")
            return True  # Not a failure, just informational
            
    except ImportError:
        print("✗ PyTorch not available for GPU testing")
        return False
    except Exception as e:
        print(f"✗ GPU test error: {e}")
        return False


def test_cli():
    """Test that the CLI script can be imported and basic commands work."""
    print("\nTesting CLI...")
    
    try:
        # Test that CLI script exists and is importable
        cli_path = Path("rl_bot_cli.py")
        if not cli_path.exists():
            print("✗ rl_bot_cli.py - missing")
            return False
        
        print("✓ rl_bot_cli.py exists")
        
        # Test basic CLI functionality by importing
        import subprocess
        result = subprocess.run([
            sys.executable, "rl_bot_cli.py", "--help"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ CLI help command works")
            return True
        else:
            print(f"✗ CLI help command failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ CLI test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("RL Bot System Setup Verification")
    print("=" * 40)
    
    tests = [
        ("Imports", test_imports),
        ("Directories", test_directories), 
        ("Configuration Files", test_config_files),
        ("GPU Availability", test_gpu_availability),
        ("CLI", test_cli)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            traceback.print_exc()
            results.append((test_name, False))
    
    print("\n" + "=" * 40)
    print("Test Results:")
    
    all_passed = True
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("✓ All tests passed! RL Bot System setup is complete.")
        return 0
    else:
        print("✗ Some tests failed. Please check the setup.")
        return 1


if __name__ == "__main__":
    sys.exit(main())