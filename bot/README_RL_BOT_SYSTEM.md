# RL Bot System

A reinforcement learning bot system for training successive generations of intelligent game-playing agents.

## Overview

This system implements a comprehensive RL training framework that creates increasingly sophisticated bot players through successive generations. Each new generation learns from and improves upon previous models, creating a progressive improvement in bot performance.

## Features

- **Successive Learning**: Train multiple RL model generations where each improves upon previous ones
- **Multiple Algorithms**: Support for DQN, PPO, A3C, and SAC algorithms
- **Flexible State Representations**: Raw coordinates, grid-based, feature vectors, and hybrid representations
- **Configurable Action Spaces**: Discrete, continuous, and hybrid action spaces
- **Advanced Reward Functions**: Multiple reward function types with multi-objective optimization
- **Cohort Training**: Train against multiple previous generations simultaneously
- **Rules-Based Foundation**: Start with intelligent rules-based bots as baseline opponents
- **Comprehensive Evaluation**: Statistical comparison and performance tracking across generations
- **Configuration Management**: YAML-based configuration system with templates and validation

## Requirements

- **Python 3.12** (required for optimal CUDA support and compatibility)
- **uv** (recommended for faster package management) - Install from: https://docs.astral.sh/uv/getting-started/installation/
- **NVIDIA GPU** (optional, for GPU acceleration)
- **CUDA 11.8 or 12.1** (optional, for GPU acceleration)

## Quick Start

### 1. Prerequisites

This project requires `uv` (a fast Python package manager) and Python 3.12:

```bash
# Install uv (if not already installed)
# Windows (PowerShell):
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Alternative (with pip):
pip install uv

# Install Python 3.12 (if needed)
uv python install 3.12
```

### 2. Setup Environment

```bash
# Navigate to the bot directory
cd bot

# Run the setup script (automatically creates Python 3.12 venv with uv)
python run_setup.py

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On Unix/macOS:
source .venv/bin/activate

# Configure GPU support (run after activating the environment)
python run_setup_gpu.py

# Test the installation
python run_setup_test.py
```

### 3. Initialize the System

```bash
# Set up directories and default configuration
python rl_bot_cli.py setup
```

### 4. View Available Configurations

```bash
# List all configurations
python rl_bot_cli.py config list

# Show default configuration details
python rl_bot_cli.py config show default

# Validate configuration
python rl_bot_cli.py config validate default
```

### 5. Create Custom Configuration

```bash
# Create a quick training configuration for experimentation
python rl_bot_cli.py config create my_experiment --template quick_training --algorithm DQN

# Create algorithm-specific configuration
python rl_bot_cli.py config create ppo_config --algorithm PPO
```

## Directory Structure

```
bot/
├── rl_bot_system/           # Main RL bot system package
│   ├── config/              # Configuration management
│   ├── models/              # RL model implementations
│   ├── training/            # Training engine and algorithms
│   ├── evaluation/          # Model evaluation and comparison
│   ├── environment/         # Game environment wrapper
│   ├── rules_based/         # Rules-based bot foundation
│   ├── storage/             # Model and data persistence
│   ├── server/              # Bot server for game integration
│   └── utils/               # Utility functions
├── data/                    # Data storage
│   ├── models/              # Trained model files
│   ├── replays/             # Game replay data
│   └── metrics/             # Training and evaluation metrics
├── logs/                    # Log files and tensorboard data
├── config/                  # Configuration files
│   ├── default.yaml         # Default configuration
│   └── quick_training.yaml  # Quick training configuration
├── game_client.py           # Existing game client
├── example_bot.py           # Example bot implementation
├── rl_bot_cli.py           # Command-line interface
├── setup.py                # Environment setup script
└── requirements.txt        # Python dependencies
```

## Configuration System

The system uses YAML configuration files to manage all aspects of training and evaluation:

### Key Configuration Sections

- **game**: Game connection and environment settings
- **state_representation**: How game state is encoded for the RL model
- **action_space**: Available actions and their encoding
- **reward**: Reward function configuration and multi-objective optimization
- **training**: RL algorithm parameters and training settings
- **cohort_training**: Settings for training against previous generations
- **evaluation**: Model evaluation and comparison settings
- **model**: Neural network architecture and model management
- **speed_control**: Game speed acceleration for faster training
- **logging**: Logging, monitoring, and visualization settings

### Example Configuration Usage

```bash
# Use default configuration
python rl_bot_cli.py --config default train start

# Use quick training configuration for experimentation
python rl_bot_cli.py --config quick_training train start

# Create and use custom configuration
python rl_bot_cli.py config create my_config --algorithm PPO
python rl_bot_cli.py --config my_config train start
```

### GPU Configuration

The system automatically detects and uses GPU acceleration when available. You can control GPU usage through configuration:

```yaml
training:
  device: "auto"  # "auto", "cpu", "cuda", "cuda:0"
  # auto: automatically use GPU if available, fallback to CPU
  # cpu: force CPU training
  # cuda: use first available GPU
  # cuda:0: use specific GPU device
```

**GPU-Optimized Settings:**
```bash
# Create GPU-optimized configuration
python rl_bot_cli.py config create gpu_training --template default

# Then edit config/gpu_training.yaml to increase batch sizes:
# training:
#   batch_size: 256        # Increase from 64
#   buffer_size: 1000000   # Increase from 100000
#   n_steps: 4096          # Increase from 2048 (for PPO)
```

## Command-Line Interface

The `rl_bot_cli.py` provides a comprehensive CLI for managing the RL bot system:

### Configuration Commands
```bash
python rl_bot_cli.py config list                    # List configurations
python rl_bot_cli.py config show [config_name]      # Show configuration details
python rl_bot_cli.py config validate [config_name]  # Validate configuration
python rl_bot_cli.py config create <name> --template <template> --algorithm <algo>
```

### Training Commands
```bash
python rl_bot_cli.py train start                    # Start training
python rl_bot_cli.py train start --generation 2     # Train specific generation
python rl_bot_cli.py train start --resume           # Resume from checkpoint
python rl_bot_cli.py train status                   # Check training status
```

### Model Management
```bash
python rl_bot_cli.py model list                     # List trained models
python rl_bot_cli.py model info <model_name>        # Show model information
```

### Evaluation Commands
```bash
python rl_bot_cli.py evaluate model <model_name>    # Evaluate a model
python rl_bot_cli.py evaluate model <model_name> --episodes 50
```

### System Commands
```bash
python rl_bot_cli.py setup                          # Set up the system
python rl_bot_cli.py setup --force                  # Force re-setup
python rl_bot_cli.py version                        # Show version
python rl_bot_cli.py gpu-info                       # Show GPU information
```

### GPU-Specific Commands
```bash
# Create GPU-optimized configuration
python rl_bot_cli.py config create my_gpu_config --gpu --algorithm PPO

# Check GPU availability and specs
python rl_bot_cli.py gpu-info

# Setup with GPU support
python setup.py --gpu
```

## Dependencies

### System Requirements

- **Python 3.12**: Required for optimal CUDA support and package compatibility
- **uv**: Fast Python package manager (required) - https://docs.astral.sh/uv/
- **NVIDIA GPU**: Optional, for GPU acceleration

### Python Packages

- **PyTorch**: Deep learning framework (with CUDA support for GPU acceleration)
- **Stable-Baselines3**: RL algorithms implementation
- **Gymnasium**: RL environment interface
- **NumPy**: Numerical computing
- **PyYAML**: Configuration file handling
- **TensorBoard**: Training visualization
- **Weights & Biases**: Experiment tracking (optional)

See `requirements.txt` for the complete list of dependencies.

### Why uv?

This project requires `uv` for package management because it provides:
- **10-100x faster** package installation compared to pip
- **Better dependency resolution** and conflict detection
- **Reliable virtual environment** creation and management
- **Cross-platform consistency** for development teams
- **Python version management** with `uv python install`

### Why Python 3.12?

Python 3.12 is required because:
- **CUDA Compatibility**: Best support for PyTorch with CUDA 11.8/12.1
- **Performance**: Improved performance for ML workloads
- **Package Compatibility**: All dependencies are tested and optimized for Python 3.12
- **Stability**: Mature ecosystem with stable package versions

### GPU Acceleration

For optimal training performance, GPU acceleration is highly recommended:

**Requirements:**
- **Python 3.12** (recommended for best CUDA compatibility)
- NVIDIA GPU with CUDA Compute Capability 3.5 or higher
- NVIDIA drivers (latest recommended)
- CUDA toolkit 11.8 or 12.1 (optional - PyTorch includes CUDA runtime)

**Setup:**
```bash
# Install Python 3.12 (recommended)
# Windows: winget install Python.Python.3.12
# Ubuntu: sudo apt install python3.12
# macOS: brew install python@3.12

# Create environment with Python 3.12
python setup.py --python python3.12 --gpu

# Check if GPU is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Or manually install CUDA-enabled PyTorch
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Performance Benefits:**
- 10-50x faster training depending on model size
- Larger batch sizes for better gradient estimates
- Ability to train more complex neural network architectures
- Faster evaluation and inference

**Memory Requirements:**
- Minimum 4GB GPU memory for basic training
- 8GB+ recommended for larger models and batch sizes
- 16GB+ for advanced architectures and parallel training

## Next Steps

This is the foundation setup for the RL bot system. The following components will be implemented in subsequent tasks:

1. **Rules-Based Bot Foundation** - Intelligent baseline bots with configurable difficulty
2. **Game Environment Wrapper** - Adapt the existing GameClient for RL training
3. **RL Training Engine** - Core training loop with successive learning
4. **Model Management** - Model lifecycle and knowledge transfer
5. **Evaluation Framework** - Performance comparison and validation
6. **Game Speed Controller** - Accelerated training with spectator support
7. **Bot Server Integration** - Player bot management and game integration

## GPU Troubleshooting

### Common Issues

**CUDA not available:**
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Reinstall PyTorch with CUDA support
uv pip uninstall torch torchvision torchaudio
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Out of memory errors:**
```bash
# Reduce batch size in configuration
python rl_bot_cli.py config show default
# Edit config file to reduce batch_size, n_steps, or buffer_size
```

**Mixed GPU/CPU environments:**
```bash
# Force CPU training if GPU causes issues
python rl_bot_cli.py config create cpu_training --template default
# Then edit config/cpu_training.yaml to set device: "cpu"
```

### Performance Monitoring

```bash
# Monitor GPU usage during training
nvidia-smi -l 1

# Check GPU memory usage in Python
python -c "
import torch
if torch.cuda.is_available():
    print(f'GPU Memory: {torch.cuda.memory_allocated()/1e9:.1f}GB / {torch.cuda.memory_reserved()/1e9:.1f}GB')
"
```

## Contributing

When implementing new features:

1. Follow the existing directory structure
2. Add appropriate configuration options to the config system
3. Update the CLI with relevant commands
4. Add proper logging and error handling
5. Include unit tests for new components
6. Update documentation
7. Test on both GPU and CPU environments when possible

## License

This project is part of the larger game system and follows the same licensing terms.