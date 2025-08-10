"""
EXPLAINIUM - Model Optimization and Caching System

Optimized for Apple M4 Mac with memory management, caching, and Apple Metal acceleration.
"""

import os
import logging
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import hashlib
import time
from datetime import datetime, timedelta
import psutil

# Apple-specific optimizations
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.logging_config import get_logger

logger = get_logger(__name__)


class DiskCache:
    """Disk-based caching system for models and embeddings"""
    
    def __init__(self, cache_dir: str, size_limit: str = "4GB"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Parse size limit
        self.size_limit = self._parse_size_limit(size_limit)
        self.cache_index_file = self.cache_dir / "cache_index.json"
        self.cache_index = self._load_cache_index()
        
        # Clean up old cache entries
        self._cleanup_cache()
    
    def _parse_size_limit(self, size_limit: str) -> int:
        """Parse human-readable size limit to bytes"""
        size_limit = size_limit.upper()
        if size_limit.endswith('GB'):
            return int(float(size_limit[:-2]) * 1024 * 1024 * 1024)
        elif size_limit.endswith('MB'):
            return int(float(size_limit[:-2]) * 1024 * 1024)
        elif size_limit.endswith('KB'):
            return int(float(size_limit[:-2]) * 1024)
        else:
            return int(size_limit)
    
    def _load_cache_index(self) -> Dict[str, Any]:
        """Load cache index from disk"""
        if self.cache_index_file.exists():
            try:
                with open(self.cache_index_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
        
        return {}
    
    def _save_cache_index(self):
        """Save cache index to disk"""
        try:
            with open(self.cache_index_file, 'w') as f:
                json.dump(self.cache_index, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")
    
    def _cleanup_cache(self):
        """Clean up old cache entries and enforce size limit"""
        current_size = 0
        current_time = time.time()
        
        # Calculate current cache size and remove expired entries
        valid_entries = {}
        for key, entry in self.cache_index.items():
            cache_file = self.cache_dir / entry['filename']
            
            # Remove expired entries
            if current_time - entry['created_at'] > entry['ttl']:
                if cache_file.exists():
                    cache_file.unlink()
                continue
            
            # Calculate size for valid entries
            if cache_file.exists():
                size = cache_file.stat().st_size
                current_size += size
                entry['size'] = size
                valid_entries[key] = entry
        
        self.cache_index = valid_entries
        
        # If still over limit, remove oldest entries
        if current_size > self.size_limit:
            sorted_entries = sorted(
                self.cache_index.items(),
                key=lambda x: x[1]['created_at']
            )
            
            for key, entry in sorted_entries:
                cache_file = self.cache_dir / entry['filename']
                if cache_file.exists():
                    cache_file.unlink()
                    current_size -= entry['size']
                    del self.cache_index[key]
                    
                    if current_size <= self.size_limit:
                        break
        
        self._save_cache_index()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        if key not in self.cache_index:
            return None
        
        entry = self.cache_index[key]
        cache_file = self.cache_dir / entry['filename']
        
        if not cache_file.exists():
            del self.cache_index[key]
            return None
        
        # Check if expired
        if time.time() - entry['created_at'] > entry['ttl']:
            cache_file.unlink()
            del self.cache_index[key]
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read cache file: {e}")
            return None
    
    def set(self, key: str, value: bytes, ttl: int = 3600):
        """Set item in cache with TTL in seconds"""
        # Generate filename from key
        filename = hashlib.md5(key.encode()).hexdigest()
        
        # Save to disk
        cache_file = self.cache_dir / filename
        try:
            with open(cache_file, 'wb') as f:
                f.write(value)
        except Exception as e:
            logger.error(f"Failed to write cache file: {e}")
            return False
        
        # Update index
        self.cache_index[key] = {
            'filename': filename,
            'created_at': time.time(),
            'ttl': ttl,
            'size': len(value)
        }
        
        self._save_cache_index()
        return True
    
    def delete(self, key: str):
        """Delete item from cache"""
        if key in self.cache_index:
            entry = self.cache_index[key]
            cache_file = self.cache_dir / entry['filename']
            
            if cache_file.exists():
                cache_file.unlink()
            
            del self.cache_index[key]
            self._save_cache_index()
    
    def clear(self):
        """Clear all cache entries"""
        for entry in self.cache_index.values():
            cache_file = self.cache_dir / entry['filename']
            if cache_file.exists():
                cache_file.unlink()
        
        self.cache_index = {}
        self._save_cache_index()


class ModelOptimizer:
    """M4-specific model optimization and management"""
    
    def __init__(self, cache_dir: str = "./model_cache"):
        self.cache = DiskCache(cache_dir, size_limit="4GB")
        self.loaded_models = {}
        self.model_usage = {}
        self.max_memory = 14 * 1024 * 1024 * 1024  # 14GB
        
    async def optimize_for_m4(self):
        """Apply M4-specific optimizations"""
        optimizations = []
        
        # Check if we can use Apple Metal
        if TORCH_AVAILABLE:
            try:
        if torch.backends.mps.is_available():
                    torch.backends.mps.enable()
                    optimizations.append("Apple Metal enabled")
                else:
                    optimizations.append("Apple Metal not available")
            except Exception as e:
                logger.warning(f"Failed to enable Apple Metal: {e}")
        
        # Check MLX availability
        if MLX_AVAILABLE:
            optimizations.append("MLX framework available")
        
        # Set optimal thread count for M4
        optimal_threads = min(8, os.cpu_count())
        os.environ['OMP_NUM_THREADS'] = str(optimal_threads)
        optimizations.append(f"Thread count optimized: {optimal_threads}")
        
        # Memory optimization
        self._optimize_memory_settings()
        optimizations.append("Memory settings optimized")
        
        logger.info(f"M4 optimizations applied: {', '.join(optimizations)}")
        return optimizations
    
    def _optimize_memory_settings(self):
        """Optimize memory settings for M4 Mac"""
        # Set environment variables for optimal performance
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.8'
        
        # Optimize NumPy if available
        try:
            import numpy as np
            np.set_printoptions(precision=4, suppress=True)
        except ImportError:
            pass
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss': memory_info.rss,  # Resident Set Size
            'vms': memory_info.vms,  # Virtual Memory Size
            'percent': process.memory_percent(),
            'available': psutil.virtual_memory().available,
            'total': psutil.virtual_memory().total
        }
    
    def should_swap_model(self, model_name: str) -> bool:
        """Determine if a model should be swapped out due to memory pressure"""
        memory_usage = self.get_memory_usage()
        
        # If using more than 80% of available memory, consider swapping
        if memory_usage['rss'] > self.max_memory * 0.8:
            return True
        
        return False
    
    async def load_model_with_fallback(self, model_name: str, model_loader_func) -> Any:
        """Load model with automatic fallback to lighter versions"""
        try:
            # Try to load the requested model
            model = await model_loader_func(model_name)
            self.loaded_models[model_name] = {
                'model': model,
                'loaded_at': time.time(),
                'usage_count': 0
            }
            return model
            
        except Exception as e:
            logger.warning(f"Failed to load model {model_name}: {e}")
            
            # Try fallback models
            fallback_models = [
                "microsoft/phi-2",
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            ]
            
            for fallback in fallback_models:
                try:
                    logger.info(f"Trying fallback model: {fallback}")
                    model = await model_loader_func(fallback)
                    self.loaded_models[fallback] = {
                        'model': model,
                        'loaded_at': time.time(),
                        'usage_count': 0,
                        'is_fallback': True
                    }
                    return model
                    
                except Exception as fallback_error:
                    logger.warning(f"Fallback model {fallback} also failed: {fallback_error}")
                    continue
            
            raise Exception("All model loading attempts failed")
    
    def track_model_usage(self, model_name: str):
        """Track model usage for optimization decisions"""
        if model_name in self.loaded_models:
            self.loaded_models[model_name]['usage_count'] += 1
            self.loaded_models[model_name]['last_used'] = time.time()
    
    def get_least_used_model(self) -> Optional[str]:
        """Get the least recently used model for potential swapping"""
        if not self.loaded_models:
            return None
        
        return min(
            self.loaded_models.keys(),
            key=lambda k: self.loaded_models[k].get('last_used', 0)
        )
    
    async def cleanup_unused_models(self):
        """Clean up unused models to free memory"""
        current_time = time.time()
        models_to_remove = []
        
        for model_name, model_info in self.loaded_models.items():
            # Remove models not used in the last hour
            if current_time - model_info.get('last_used', 0) > 3600:
                models_to_remove.append(model_name)
        
        for model_name in models_to_remove:
            try:
                del self.loaded_models[model_name]
                logger.info(f"Removed unused model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to remove model {model_name}: {e}")
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        return {
            'loaded_models': len(self.loaded_models),
            'cache_size': len(self.cache.cache_index),
            'memory_usage': self.get_memory_usage(),
            'model_usage': {
                name: info['usage_count'] 
                for name, info in self.loaded_models.items()
            }
        }


class StreamingProcessor:
    """Streaming document processing for large files"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            
            # Ensure we don't get stuck
            if start >= len(text):
                break
        
        return chunks

    async def process_streaming(self, text: str, processor_func) -> List[Any]:
        """Process text in streaming fashion"""
        chunks = self.chunk_text(text)
        results = []
        
        for i, chunk in enumerate(chunks):
            try:
                result = await processor_func(chunk, i, len(chunks))
                results.append(result)
                
                # Yield progress
                if hasattr(processor_func, 'progress_callback'):
                    progress = (i + 1) / len(chunks) * 100
                    processor_func.progress_callback(progress)
                    
            except Exception as e:
                logger.error(f"Error processing chunk {i}: {e}")
                results.append({'error': str(e), 'chunk_index': i})
        
        return results


class PerformanceMonitor:
    """Monitor and optimize performance metrics"""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = time.time()
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.metrics[operation] = {
            'start_time': time.time(),
            'status': 'running'
        }
    
    def end_timer(self, operation: str):
        """End timing an operation"""
        if operation in self.metrics:
            self.metrics[operation]['end_time'] = time.time()
            self.metrics[operation]['duration'] = (
                self.metrics[operation]['end_time'] - 
                self.metrics[operation]['start_time']
            )
            self.metrics[operation]['status'] = 'completed'
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report"""
        total_time = time.time() - self.start_time
        
        return {
            'total_runtime': total_time,
            'operations': self.metrics,
            'summary': {
                'total_operations': len(self.metrics),
                'completed_operations': len([
                    op for op in self.metrics.values() 
                    if op.get('status') == 'completed'
                ]),
                'average_duration': sum(
                    op.get('duration', 0) 
                    for op in self.metrics.values() 
                    if op.get('status') == 'completed'
                ) / max(len([
                    op for op in self.metrics.values() 
                    if op.get('status') == 'completed'
                ]), 1)
            }
        }