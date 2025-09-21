#!/usr/bin/env python3
"""
Simple syntax checker for the RL environment implementation.
"""

import ast
import sys
from pathlib import Path

def check_syntax(file_path):
    """Check if a Python file has valid syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        ast.parse(source)
        return True, None
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Error reading file: {e}"

def main():
    """Check syntax of all Python files in the environment module."""
    env_dir = Path("rl_bot_system/environment")
    
    if not env_dir.exists():
        print(f"Directory {env_dir} not found!")
        return 1
    
    python_files = list(env_dir.rglob("*.py"))
    
    if not python_files:
        print("No Python files found!")
        return 1
    
    print(f"Checking syntax of {len(python_files)} Python files...")
    
    errors = []
    for file_path in python_files:
        is_valid, error = check_syntax(file_path)
        if is_valid:
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path}: {error}")
            errors.append((file_path, error))
    
    if errors:
        print(f"\n{len(errors)} files have syntax errors:")
        for file_path, error in errors:
            print(f"  {file_path}: {error}")
        return 1
    else:
        print(f"\nAll {len(python_files)} files have valid syntax!")
        return 0

if __name__ == "__main__":
    sys.exit(main())