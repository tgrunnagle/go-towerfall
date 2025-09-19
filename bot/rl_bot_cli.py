#!/usr/bin/env python3
"""
Command-line interface for the RL Bot System.
Provides commands for training, evaluation, and management of RL bots.
"""
import click
import sys
from pathlib import Path

# Add the current directory to Python path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from rl_bot_system.config.config_manager import config_manager


@click.group()
@click.option('--config', '-c', default='default', help='Configuration name to use')
@click.pass_context
def cli(ctx, config):
    """RL Bot System - Train and manage reinforcement learning bots."""
    ctx.ensure_object(dict)
    ctx.obj['config_name'] = config
    
    # Load configuration
    try:
        ctx.obj['config'] = config_manager.load_config(config)
    except FileNotFoundError:
        click.echo(f"Error: Configuration '{config}' not found.", err=True)
        click.echo(f"Available configurations: {', '.join(config_manager.list_configs())}")
        sys.exit(1)


@cli.group()
def config():
    """Manage configuration files."""
    pass


@config.command('list')
def list_configs():
    """List all available configurations."""
    configs = config_manager.list_configs()
    if not configs:
        click.echo("No configurations found.")
        return
    
    click.echo("Available configurations:")
    for config_name in configs:
        click.echo(f"  - {config_name}")


@config.command('show')
@click.argument('config_name', required=False)
@click.pass_context
def show_config(ctx, config_name):
    """Show configuration details."""
    if not config_name:
        config_name = ctx.obj['config_name']
    
    try:
        config = config_manager.load_config(config_name)
        click.echo(f"Configuration: {config_name}")
        click.echo("=" * 40)
        
        # Show key configuration details
        click.echo(f"Algorithm: {config.training.algorithm}")
        click.echo(f"Total timesteps: {config.training.total_timesteps:,}")
        click.echo(f"Learning rate: {config.training.learning_rate}")
        click.echo(f"State representation: {config.state_representation.representation_type}")
        click.echo(f"Action space: {config.action_space.action_type}")
        click.echo(f"Reward function: {config.reward.primary_function}")
        click.echo(f"Models directory: {config.models_dir}")
        
    except FileNotFoundError:
        click.echo(f"Error: Configuration '{config_name}' not found.", err=True)


@config.command('validate')
@click.argument('config_name', required=False)
@click.pass_context
def validate_config(ctx, config_name):
    """Validate a configuration file."""
    if not config_name:
        config_name = ctx.obj['config_name']
    
    try:
        config = config_manager.load_config(config_name)
        config.validate()
        click.echo(f"✓ Configuration '{config_name}' is valid.")
    except Exception as e:
        click.echo(f"✗ Configuration '{config_name}' validation failed: {e}", err=True)
        sys.exit(1)


@config.command('create')
@click.argument('config_name')
@click.option('--template', '-t', default='default', help='Template configuration to base on')
@click.option('--algorithm', '-a', help='RL algorithm to use')
@click.option('--gpu', is_flag=True, help='Optimize for GPU training')
def create_config(config_name, template, algorithm, gpu):
    """Create a new configuration from a template."""
    try:
        overrides = {}
        
        if algorithm:
            overrides['training'] = {'algorithm': algorithm}
        
        if gpu:
            # Use GPU template or add GPU optimizations
            if template == 'default':
                template = 'gpu_training'
            
            # Add GPU-specific overrides
            gpu_overrides = {
                'training': {
                    'device': 'auto',
                    'batch_size': 256,
                    'n_steps': 4096,
                    'buffer_size': 1000000
                },
                'model': {
                    'hidden_layers': [512, 512, 256]
                },
                'speed_control': {
                    'max_parallel_episodes': 8,
                    'batch_episode_size': 32
                }
            }
            
            # Merge with existing overrides
            for key, value in gpu_overrides.items():
                if key in overrides:
                    overrides[key].update(value)
                else:
                    overrides[key] = value
        
        config = config_manager.create_config_from_template(template, config_name, overrides)
        click.echo(f"✓ Created configuration '{config_name}' based on '{template}'")
        
        if algorithm:
            click.echo(f"  - Set algorithm to {algorithm}")
        
        if gpu:
            click.echo(f"  - Optimized for GPU training")
            
    except Exception as e:
        click.echo(f"Error creating configuration: {e}", err=True)
        sys.exit(1)


@cli.group()
def train():
    """Training commands."""
    pass


@train.command('start')
@click.option('--generation', '-g', type=int, help='Generation number to train')
@click.option('--resume', '-r', is_flag=True, help='Resume training from checkpoint')
@click.pass_context
def start_training(ctx, generation, resume):
    """Start training a new RL model generation."""
    config = ctx.obj['config']
    
    click.echo("Starting RL bot training...")
    click.echo(f"Configuration: {ctx.obj['config_name']}")
    click.echo(f"Algorithm: {config.training.algorithm}")
    click.echo(f"Total timesteps: {config.training.total_timesteps:,}")
    
    if generation:
        click.echo(f"Training generation: {generation}")
    
    if resume:
        click.echo("Resuming from checkpoint...")
    
    # TODO: Implement actual training logic
    click.echo("Training implementation coming in future tasks...")


@train.command('status')
@click.pass_context
def training_status(ctx):
    """Show current training status."""
    click.echo("Checking training status...")
    # TODO: Implement training status check
    click.echo("Status check implementation coming in future tasks...")


@cli.group()
def model():
    """Model management commands."""
    pass


@model.command('list')
@click.pass_context
def list_models(ctx):
    """List all trained models."""
    config = ctx.obj['config']
    models_dir = Path(config.models_dir)
    
    if not models_dir.exists():
        click.echo("No models directory found.")
        return
    
    model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
    
    if not model_dirs:
        click.echo("No trained models found.")
        return
    
    click.echo("Trained models:")
    for model_dir in sorted(model_dirs):
        click.echo(f"  - {model_dir.name}")


@model.command('info')
@click.argument('model_name')
@click.pass_context
def model_info(ctx, model_name):
    """Show information about a specific model."""
    config = ctx.obj['config']
    model_path = Path(config.models_dir) / model_name
    
    if not model_path.exists():
        click.echo(f"Error: Model '{model_name}' not found.", err=True)
        return
    
    click.echo(f"Model: {model_name}")
    click.echo("=" * 40)
    
    # Check for common files
    files = list(model_path.iterdir())
    click.echo(f"Files: {len(files)}")
    for file in files:
        click.echo(f"  - {file.name}")


@cli.group()
def evaluate():
    """Evaluation commands."""
    pass


@evaluate.command('model')
@click.argument('model_name')
@click.option('--episodes', '-e', default=10, help='Number of evaluation episodes')
@click.pass_context
def evaluate_model(ctx, model_name, episodes):
    """Evaluate a trained model."""
    config = ctx.obj['config']
    
    click.echo(f"Evaluating model: {model_name}")
    click.echo(f"Episodes: {episodes}")
    click.echo(f"Configuration: {ctx.obj['config_name']}")
    
    # TODO: Implement model evaluation
    click.echo("Evaluation implementation coming in future tasks...")


@cli.command('setup')
@click.option('--force', is_flag=True, help='Force setup even if already configured')
def setup_system(force):
    """Set up the RL bot system environment."""
    click.echo("Setting up RL Bot System...")
    
    # Create directories
    directories = ['data/models', 'data/replays', 'data/metrics', 'logs', 'config']
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        click.echo(f"✓ Created directory: {directory}")
    
    # Create default configuration if it doesn't exist
    if force or 'default' not in config_manager.list_configs():
        config = config_manager.get_default_config()
        click.echo("✓ Created default configuration")
    else:
        click.echo("✓ Default configuration already exists")
    
    click.echo("Setup completed successfully!")


@cli.command('gpu-info')
def gpu_info():
    """Show GPU information and availability."""
    try:
        import torch
        
        click.echo("GPU Information:")
        click.echo("=" * 40)
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            cuda_version = torch.version.cuda
            
            click.echo(f"CUDA Available: Yes (v{cuda_version})")
            click.echo(f"GPU Count: {gpu_count}")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_props = torch.cuda.get_device_properties(i)
                gpu_memory = gpu_props.total_memory / (1024**3)
                
                click.echo(f"\nGPU {i}: {gpu_name}")
                click.echo(f"  Memory: {gpu_memory:.1f} GB")
                click.echo(f"  Compute Capability: {gpu_props.major}.{gpu_props.minor}")
                
                # Check current memory usage
                if torch.cuda.is_initialized():
                    allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    reserved = torch.cuda.memory_reserved(i) / (1024**3)
                    click.echo(f"  Memory Used: {allocated:.1f} GB / {reserved:.1f} GB reserved")
        else:
            click.echo("CUDA Available: No")
            click.echo("Reasons CUDA might not be available:")
            click.echo("  - No NVIDIA GPU installed")
            click.echo("  - NVIDIA drivers not installed")
            click.echo("  - CUDA toolkit not installed")
            click.echo("  - PyTorch installed without CUDA support")
            
    except ImportError:
        click.echo("Error: PyTorch not installed")
    except Exception as e:
        click.echo(f"Error checking GPU: {e}")


@cli.command('version')
def version():
    """Show version information."""
    click.echo("RL Bot System v0.1.0")
    
    # Also show PyTorch and CUDA versions
    try:
        import torch
        click.echo(f"PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            click.echo(f"CUDA: {torch.version.cuda}")
        else:
            click.echo("CUDA: Not available")
    except ImportError:
        click.echo("PyTorch: Not installed")


if __name__ == '__main__':
    cli()