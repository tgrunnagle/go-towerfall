#!/usr/bin/env python3
"""
Test runner script for the bot project.
Handles pytest execution with proper path setup and import resolution.
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path


def setup_python_path():
    """Add the bot directory to Python path for proper imports"""
    bot_dir = Path(__file__).parent.absolute()
    if str(bot_dir) not in sys.path:
        sys.path.insert(0, str(bot_dir))
    
    # Also set PYTHONPATH environment variable
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    if str(bot_dir) not in current_pythonpath:
        if current_pythonpath:
            os.environ['PYTHONPATH'] = f"{bot_dir}{os.pathsep}{current_pythonpath}"
        else:
            os.environ['PYTHONPATH'] = str(bot_dir)


def find_test_files(path):
    """Find all test files in the given path"""
    path = Path(path)
    if path.is_file() and path.name.startswith('test_') and path.suffix == '.py':
        return [path]
    elif path.is_dir():
        test_files = []
        for test_file in path.rglob('test_*.py'):
            # Skip files in .venv, __pycache__, and other common directories to ignore
            if any(part.startswith('.') or part == '__pycache__' or part == 'node_modules' 
                   for part in test_file.parts):
                continue
            test_files.append(test_file)
        return test_files
    else:
        return []


def run_pytest(test_paths, verbose=True, capture=False):
    """Run pytest with the given test paths"""
    setup_python_path()
    
    # Use uv to run pytest in the virtual environment
    cmd = ['uv', 'run', 'pytest']
    
    if verbose:
        cmd.append('-v')
    
    if not capture:
        cmd.append('-s')  # Don't capture output
    
    # Add test paths
    for path in test_paths:
        cmd.append(str(path))
    
    # Run from bot directory
    bot_dir = Path(__file__).parent
    
    print(f"Running: {' '.join(cmd)}")
    print(f"Working directory: {bot_dir}")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    print("-" * 50)
    
    result = subprocess.run(cmd, cwd=bot_dir)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description='Run tests for the bot project')
    parser.add_argument(
        'target', 
        nargs='?', 
        default='.',
        help='Test file, directory, or "." for all tests in bot directory'
    )
    parser.add_argument(
        '-v', '--verbose', 
        action='store_true', 
        default=True,
        help='Verbose output (default: True)'
    )
    parser.add_argument(
        '-q', '--quiet', 
        action='store_true',
        help='Quiet output (overrides verbose)'
    )
    parser.add_argument(
        '-c', '--capture', 
        action='store_true',
        help='Capture output (don\'t show print statements)'
    )
    
    args = parser.parse_args()
    
    # Handle quiet flag
    verbose = args.verbose and not args.quiet
    
    # Resolve target path
    bot_dir = Path(__file__).parent
    if args.target == '.':
        target_path = bot_dir
    else:
        target_path = bot_dir / args.target
        if not target_path.exists():
            # Try as absolute path
            target_path = Path(args.target)
            if not target_path.exists():
                print(f"Error: Target path '{args.target}' does not exist")
                return 1
    
    # Find test files
    test_files = find_test_files(target_path)
    
    if not test_files:
        print(f"No test files found in '{target_path}'")
        return 1
    
    print(f"Found {len(test_files)} test file(s):")
    for test_file in test_files:
        print(f"  - {test_file.relative_to(bot_dir)}")
    print()
    
    # Run tests
    return run_pytest(test_files, verbose=verbose, capture=args.capture)


if __name__ == '__main__':
    sys.exit(main())