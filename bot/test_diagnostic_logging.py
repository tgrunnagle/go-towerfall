#!/usr/bin/env python3
"""
Test script to verify BOT_DIAGNOSTIC events only go to log files, not console.
"""

import sys
import os
import logging
from datetime import datetime

# Add the server directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'server'))

from diagnostics import BotDiagnosticTracker, BotLifecycleEvent, DiagnosticLevel

def test_diagnostic_logging():
    """Test that BOT_DIAGNOSTIC events only go to files, not console."""
    
    print("Testing BOT_DIAGNOSTIC logging behavior...")
    print("This message should appear in console.")
    
    # Create diagnostic tracker
    tracker = BotDiagnosticTracker()
    
    # Register a test bot
    tracker.register_bot("test_bot", "Test Bot")
    
    print("Registered test bot - this should appear in console.")
    
    # Log some diagnostic events - these should NOT appear in console
    tracker.log_event(
        bot_id="test_bot",
        event_type=BotLifecycleEvent.BOT_CONNECTING,
        level=DiagnosticLevel.INFO,
        message="Test connection event - should only be in log file",
        details={"test": "data"}
    )
    
    tracker.log_event(
        bot_id="test_bot",
        event_type=BotLifecycleEvent.BOT_GAME_JOINED,
        level=DiagnosticLevel.INFO,
        message="Test game joined event - should only be in log file"
    )
    
    tracker.log_event(
        bot_id="test_bot",
        event_type=BotLifecycleEvent.BOT_ERROR,
        level=DiagnosticLevel.ERROR,
        message="Test error event - should only be in log file",
        details={"error": "test error"}
    )
    
    print("Logged diagnostic events - these should NOT appear in console.")
    
    # Check if log file was created
    logs_dir = os.path.join(os.path.dirname(__file__), 'server', 'logs')
    log_file = os.path.join(logs_dir, 'bot_diagnostics.log')
    
    if os.path.exists(log_file):
        print(f"✓ Log file created: {log_file}")
        
        # Read and display log file contents
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        print("\n--- Log file contents ---")
        print(content)
        print("--- End log file contents ---")
        
        # Check if BOT_DIAGNOSTIC events are in the file
        if "BOT_DIAGNOSTIC" in content:
            print("✓ BOT_DIAGNOSTIC events found in log file")
        else:
            print("✗ BOT_DIAGNOSTIC events NOT found in log file")
            
    else:
        print(f"✗ Log file not created: {log_file}")
    
    # Clean up
    tracker.unregister_bot("test_bot")
    
    print("\nTest completed. BOT_DIAGNOSTIC events should only appear in the log file, not in this console output.")

if __name__ == "__main__":
    test_diagnostic_logging()