"""
Performance profiling utilities for the fraud detection application.
Provides timing decorators and profiling helpers for optimization tracking.
"""

import time
import functools
from typing import Callable, Any
import streamlit as st


def get_timer():
    """Get the current time in milliseconds."""
    return time.perf_counter() * 1000


def profile_function(show_time: bool = True) -> Callable:
    """
    Decorator to profile function execution time.
    
    Args:
        show_time: If True, print execution time to console
    
    Returns:
        Decorated function with timing capability
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start = get_timer()
            result = func(*args, **kwargs)
            elapsed = get_timer() - start
            
            if show_time:
                print(f"⏱️  {func.__name__}: {elapsed:.1f}ms")
            
            return result
        return wrapper
    return decorator


class PerformanceTracker:
    """
    Simple performance tracking system for monitoring operation speeds.
    Useful for identifying bottlenecks in data processing.
    """
    
    def __init__(self):
        self.timings = {}
    
    def start(self, operation_name: str):
        """Start tracking an operation."""
        self.timings[operation_name] = {"start": get_timer()}
    
    def end(self, operation_name: str) -> float:
        """
        End tracking an operation and return elapsed time in milliseconds.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Elapsed time in milliseconds
        """
        if operation_name not in self.timings:
            return 0.0
        
        elapsed = get_timer() - self.timings[operation_name]["start"]
        self.timings[operation_name]["elapsed"] = elapsed
        return elapsed
    
    def get_summary(self) -> dict:
        """Get summary of all tracked operations."""
        return {
            name: data.get("elapsed", 0.0)
            for name, data in self.timings.items()
        }
    
    def display_in_streamlit(self, location="sidebar"):
        """Display timing summary in Streamlit sidebar or main."""
        summary = self.get_summary()
        if not summary:
            return
        
        display_fn = st.sidebar if location == "sidebar" else st
        
        with display_fn.expander("⏱️ Performance Metrics"):
            total = sum(summary.values())
            st.markdown(f"**Total Processing Time:** `{total:.1f}ms`")
            st.markdown("---")
            
            for op, elapsed in sorted(summary.items(), key=lambda x: x[1], reverse=True):
                pct = (elapsed / total * 100) if total > 0 else 0
                st.caption(f"{op}: **{elapsed:.1f}ms** ({pct:.1f}%)")


# Global tracker instance
_tracker = PerformanceTracker()


def get_tracker() -> PerformanceTracker:
    """Get global performance tracker instance."""
    return _tracker
