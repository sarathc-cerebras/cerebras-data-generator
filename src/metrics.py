import time
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import asyncio
import logging

logger = logging.getLogger("uvicorn.error")

class MetricsTracker:
    """Track system metrics for monitoring and analytics."""
    
    def __init__(self, window_seconds: int = 60):
        self.window_seconds = window_seconds
        self.requests_sent = deque()  # (timestamp, model_name)
        self.requests_completed = deque()  # (timestamp, model_name, duration)
        self.requests_failed = deque()  # (timestamp, model_name, error)
        self.rate_limits = deque()  # (timestamp, model_name)
        self.concurrency_changes = deque()  # (timestamp, model_name, old_val, new_val)
        
        # Model-specific metrics
        self.model_stats: Dict[str, Dict[str, Any]] = {}
        
        # System start time
        self.start_time = time.time()
        
        # Lock for thread safety
        self.lock = asyncio.Lock()
    
    def _cleanup_old_entries(self):
        """Remove entries older than the window."""
        cutoff = time.time() - self.window_seconds
        
        while self.requests_sent and self.requests_sent[0][0] < cutoff:
            self.requests_sent.popleft()
        
        while self.requests_completed and self.requests_completed[0][0] < cutoff:
            self.requests_completed.popleft()
        
        while self.requests_failed and self.requests_failed[0][0] < cutoff:
            self.requests_failed.popleft()
        
        while self.rate_limits and self.rate_limits[0][0] < cutoff:
            self.rate_limits.popleft()
    
    async def record_request_sent(self, model_name: str):
        """Record a request being sent."""
        async with self.lock:
            self.requests_sent.append((time.time(), model_name))
            self._cleanup_old_entries()
            
            if model_name not in self.model_stats:
                self.model_stats[model_name] = {
                    'total_sent': 0,
                    'total_completed': 0,
                    'total_failed': 0,
                    'total_duration': 0.0,
                    'rate_limits': 0
                }
            self.model_stats[model_name]['total_sent'] += 1
    
    async def record_request_completed(self, model_name: str, duration: float):
        """Record a completed request."""
        async with self.lock:
            self.requests_completed.append((time.time(), model_name, duration))
            self._cleanup_old_entries()
            
            if model_name in self.model_stats:
                self.model_stats[model_name]['total_completed'] += 1
                self.model_stats[model_name]['total_duration'] += duration
    
    async def record_request_failed(self, model_name: str, error: str):
        """Record a failed request."""
        async with self.lock:
            self.requests_failed.append((time.time(), model_name, error))
            self._cleanup_old_entries()
            
            if model_name in self.model_stats:
                self.model_stats[model_name]['total_failed'] += 1
    
    async def record_rate_limit(self, model_name: str):
        """Record a rate limit hit."""
        async with self.lock:
            self.rate_limits.append((time.time(), model_name))
            self._cleanup_old_entries()
            
            if model_name in self.model_stats:
                self.model_stats[model_name]['rate_limits'] += 1
    
    async def record_concurrency_change(self, model_name: str, old_val: int, new_val: int):
        """Record a concurrency adjustment."""
        async with self.lock:
            self.concurrency_changes.append((time.time(), model_name, old_val, new_val))
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        async with self.lock:
            self._cleanup_old_entries()
            
            # Calculate RPM (requests per minute)
            rpm_sent = len(self.requests_sent) * (60 / self.window_seconds)
            rpm_completed = len(self.requests_completed) * (60 / self.window_seconds)
            
            # Calculate average latency
            if self.requests_completed:
                avg_latency = sum(d for _, _, d in self.requests_completed) / len(self.requests_completed)
            else:
                avg_latency = 0
            
            # Calculate success rate
            total_finished = len(self.requests_completed) + len(self.requests_failed)
            success_rate = (len(self.requests_completed) / total_finished * 100) if total_finished > 0 else 0
            
            # Calculate uptime
            uptime_seconds = time.time() - self.start_time
            uptime_hours = uptime_seconds / 3600
            
            # Model-specific metrics
            model_metrics = {}
            for model_name, stats in self.model_stats.items():
                # Calculate per-model RPM
                model_sent_recent = sum(1 for t, m in self.requests_sent if m == model_name)
                model_completed_recent = sum(1 for t, m, _ in self.requests_completed if m == model_name)
                
                model_rpm = model_completed_recent * (60 / self.window_seconds)
                
                # Calculate per-model average latency
                model_durations = [d for _, m, d in self.requests_completed if m == model_name]
                model_avg_latency = sum(model_durations) / len(model_durations) if model_durations else 0
                
                # Calculate per-model success rate
                model_total = stats['total_completed'] + stats['total_failed']
                model_success_rate = (stats['total_completed'] / model_total * 100) if model_total > 0 else 0
                
                model_metrics[model_name] = {
                    'rpm': round(model_rpm, 2),
                    'avg_latency': round(model_avg_latency, 3),
                    'success_rate': round(model_success_rate, 2),
                    'total_sent': stats['total_sent'],
                    'total_completed': stats['total_completed'],
                    'total_failed': stats['total_failed'],
                    'rate_limits': stats['rate_limits'],
                    'rate_limits_recent': sum(1 for t, m in self.rate_limits if m == model_name)
                }
            
            return {
                'overview': {
                    'rpm_sent': round(rpm_sent, 2),
                    'rpm_completed': round(rpm_completed, 2),
                    'avg_latency_seconds': round(avg_latency, 3),
                    'success_rate': round(success_rate, 2),
                    'uptime_hours': round(uptime_hours, 2),
                    'total_rate_limits': len(self.rate_limits),
                    'window_seconds': self.window_seconds
                },
                'recent': {
                    'requests_sent': len(self.requests_sent),
                    'requests_completed': len(self.requests_completed),
                    'requests_failed': len(self.requests_failed),
                    'rate_limits_hit': len(self.rate_limits)
                },
                'models': model_metrics,
                'lifetime': {
                    model_name: {
                        'total_sent': stats['total_sent'],
                        'total_completed': stats['total_completed'],
                        'total_failed': stats['total_failed'],
                        'total_rate_limits': stats['rate_limits']
                    }
                    for model_name, stats in self.model_stats.items()
                }
            }