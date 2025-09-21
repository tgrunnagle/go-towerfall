"""
Async environment wrapper utilities.

This module provides utilities for handling asynchronous operations
in the RL environment, particularly for integrating with the async GameClient.
"""

import asyncio
import threading
from typing import Any, Dict, Optional
from concurrent.futures import ThreadPoolExecutor


class AsyncEnvWrapper:
    """
    Wrapper to handle async operations in synchronous RL environment.
    
    This class provides utilities to run async operations from sync contexts,
    which is necessary when integrating async GameClient with sync RL libraries.
    """
    
    def __init__(self):
        self.loop = None
        self.thread = None
        self.executor = ThreadPoolExecutor(max_workers=1)
    
    def start_async_loop(self):
        """Start the async event loop in a separate thread."""
        if self.thread is not None:
            return
        
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        
        # Wait for loop to be ready
        while self.loop is None:
            threading.Event().wait(0.01)
    
    def _run_loop(self):
        """Run the async event loop."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
    
    def run_async(self, coro):
        """
        Run an async coroutine from sync context.
        
        Args:
            coro: Coroutine to run
            
        Returns:
            Result of the coroutine
        """
        if self.loop is None:
            self.start_async_loop()
        
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result()
    
    def stop(self):
        """Stop the async loop and cleanup."""
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self.loop.stop)
        
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        
        self.executor.shutdown(wait=True)