"""
EXPLAINIUM - Apple M4 Mac Optimization Module

Performance and memory optimization specifically designed for Apple M4 Mac
with 16GB RAM. Includes model management, memory monitoring, and caching.
"""

import os
import gc
import psutil
import threading
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path
import logging
from contextlib import contextmanager

import torch
import numpy as np
from diskcache import Cache

# Apple Silicon optimization
try:
    import mlx.core as mx
    import mlx.nn as nn
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

from src.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class MemoryStats:
    """Memory usage statistics"""
    total_ram: float
    available_ram: float
    used_ram: float
    ram_percent: float
    gpu_memory: Optional[float] = None
    cache_size: float = 0.0


@dataclass
class ModelConfig:
    """Configuration for AI model optimization"""
    model_name: str
    quantization: str = "4bit"
    max_memory_mb: int = 2048
    use_cache: bool = True
    use_metal: bool = True
    batch_size: int = 4
    context_length: int = 4096


class MemoryMonitor:
    """Real-time memory monitoring and management"""
    
    def __init__(self, warning_threshold: float = 0.8, critical_threshold: float = 0.9):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.monitoring = False
        self.monitor_thread = None
        self.callbacks = []
        
    def start_monitoring(self, interval: int = 5):
        """Start memory monitoring in background thread"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("Memory monitoring stopped")
    
    def _monitor_loop(self, interval: int):
        """Memory monitoring loop"""
        while self.monitoring:
            try:
                stats = self.get_memory_stats()
                
                if stats.ram_percent > self.critical_threshold:
                    logger.critical(f"Critical memory usage: {stats.ram_percent:.1%}")
                    self._trigger_memory_cleanup()
                elif stats.ram_percent > self.warning_threshold:
                    logger.warning(f"High memory usage: {stats.ram_percent:.1%}")
                
                # Trigger callbacks
                for callback in self.callbacks:
                    callback(stats)
                    
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
            
            time.sleep(interval)
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        memory = psutil.virtual_memory()
        
        stats = MemoryStats(
            total_ram=memory.total / (1024**3),  # GB
            available_ram=memory.available / (1024**3),  # GB
            used_ram=memory.used / (1024**3),  # GB
            ram_percent=memory.percent / 100  # Fraction
        )
        
        # Get GPU memory if available
        if torch.backends.mps.is_available():
            try:
                # MPS doesn't have direct memory query, estimate based on allocated tensors
                stats.gpu_memory = torch.mps.current_allocated_memory() / (1024**3)
            except:
                stats.gpu_memory = 0.0
        
        return stats
    
    def add_callback(self, callback):
        """Add callback for memory events"""
        self.callbacks.append(callback)
    
    def _trigger_memory_cleanup(self):
        """Trigger emergency memory cleanup"""
        logger.info("Triggering emergency memory cleanup")
        gc.collect()
        
        if torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except:
                pass


class ModelManager:
    """Advanced model management with Apple M4 optimization"""
    
    def __init__(self, cache_dir: str = "./model_cache", max_cache_size: int = 4):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize disk cache (4GB limit)
        self.cache = Cache(
            directory=str(self.cache_dir),
            size_limit=max_cache_size * 1024**3  # Convert GB to bytes
        )
        
        self.loaded_models = {}
        self.model_configs = {}
        self.memory_monitor = MemoryMonitor()
        
        # Start memory monitoring
        self.memory_monitor.start_monitoring()
        self.memory_monitor.add_callback(self._on_memory_event)
        
        logger.info("ModelManager initialized with M4 optimization")
    
    def _on_memory_event(self, stats: MemoryStats):
        """Handle memory events"""
        if stats.ram_percent > 0.85:
            # Unload least recently used models
            self._unload_lru_models()
    
    def register_model_config(self, model_name: str, config: ModelConfig):
        """Register model configuration"""
        self.model_configs[model_name] = config
        logger.info(f"Registered config for model: {model_name}")
    
    @contextmanager
    def load_model(self, model_name: str, force_reload: bool = False):
        """Context manager for loading and managing models"""
        model = None
        try:
            model = self._load_model_internal(model_name, force_reload)
            yield model
        finally:
            # Optionally unload model after use to free memory
            if model and model_name in self.loaded_models:
                # Keep in memory but mark as used
                self.loaded_models[model_name]['last_used'] = time.time()
    
    def _load_model_internal(self, model_name: str, force_reload: bool = False):
        """Internal model loading with caching"""
        if not force_reload and model_name in self.loaded_models:
            self.loaded_models[model_name]['last_used'] = time.time()
            return self.loaded_models[model_name]['model']
        
        config = self.model_configs.get(model_name)
        if not config:
            raise ValueError(f"No configuration found for model: {model_name}")
        
        # Check memory before loading
        stats = self.memory_monitor.get_memory_stats()
        if stats.ram_percent > 0.8:
            logger.warning("High memory usage, cleaning up before loading model")
            self._cleanup_memory()
        
        # Load model with M4 optimization
        model = self._load_with_optimization(model_name, config)
        
        # Store in cache
        self.loaded_models[model_name] = {
            'model': model,
            'config': config,
            'loaded_at': time.time(),
            'last_used': time.time()
        }
        
        logger.info(f"Model {model_name} loaded successfully")
        return model
    
    def _load_with_optimization(self, model_name: str, config: ModelConfig):
        """Load model with Apple M4 specific optimizations"""
        try:
            if config.model_name.endswith('.gguf'):
                # Load quantized GGUF model with llama-cpp-python
                return self._load_gguf_model(config)
            else:
                # Load Hugging Face model with optimizations
                return self._load_hf_model(config)
                
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise
    
    def _load_gguf_model(self, config: ModelConfig):
        """Load GGUF model optimized for M4"""
        try:
            from llama_cpp import Llama
            
            model_path = Path("./models") / config.model_name
            
            if not model_path.exists():
                logger.warning(f"Model file not found: {model_path}")
                return None
            
            model = Llama(
                model_path=str(model_path),
                n_gpu_layers=-1 if config.use_metal else 0,  # Use all layers on Metal
                n_ctx=config.context_length,
                n_batch=config.batch_size * 128,  # Optimize batch for M4
                verbose=False,
                use_mmap=True,  # Memory mapping for efficiency
                use_mlock=True,  # Lock memory pages
                n_threads=8,  # M4 has 8 performance cores
                rope_scaling_type=1  # Optimized rope scaling
            )
            
            return model
            
        except ImportError:
            logger.error("llama-cpp-python not available")
            return None
        except Exception as e:
            logger.error(f"Error loading GGUF model: {e}")
            return None
    
    def _load_hf_model(self, config: ModelConfig):
        """Load Hugging Face model with optimizations"""
        try:
            from transformers import AutoModel, AutoTokenizer
            
            device = "mps" if torch.backends.mps.is_available() and config.use_metal else "cpu"
            
            # Load with optimizations
            model = AutoModel.from_pretrained(
                config.model_name,
                torch_dtype=torch.float16 if device == "mps" else torch.float32,
                device_map="auto" if device == "mps" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            if device == "mps":
                model = model.to(device)
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading HF model: {e}")
            return None
    
    def _unload_lru_models(self, keep_count: int = 2):
        """Unload least recently used models"""
        if len(self.loaded_models) <= keep_count:
            return
        
        # Sort by last used time
        sorted_models = sorted(
            self.loaded_models.items(),
            key=lambda x: x[1]['last_used']
        )
        
        # Unload oldest models
        unload_count = len(self.loaded_models) - keep_count
        for model_name, _ in sorted_models[:unload_count]:
            self.unload_model(model_name)
    
    def unload_model(self, model_name: str):
        """Unload specific model from memory"""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            gc.collect()
            
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
            logger.info(f"Model {model_name} unloaded")
    
    def _cleanup_memory(self):
        """Perform aggressive memory cleanup"""
        # Unload all but the most recent model
        if self.loaded_models:
            self._unload_lru_models(keep_count=1)
        
        # Clear caches
        gc.collect()
        
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        # Clear disk cache if needed
        if self.cache.volume() > 3 * 1024**3:  # > 3GB
            self.cache.clear()
            logger.info("Disk cache cleared")
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded models"""
        stats = {
            'loaded_models': len(self.loaded_models),
            'cache_size_gb': self.cache.volume() / (1024**3),
            'memory_stats': self.memory_monitor.get_memory_stats(),
            'models': {}
        }
        
        for name, info in self.loaded_models.items():
            stats['models'][name] = {
                'loaded_at': info['loaded_at'],
                'last_used': info['last_used'],
                'config': info['config'].__dict__
            }
        
        return stats
    
    def optimize_for_inference(self):
        """Apply inference-specific optimizations"""
        try:
            # Set optimal settings for M4
            torch.set_num_threads(8)  # M4 has 8 performance cores
            
            if torch.backends.mps.is_available():
                # Enable MPS optimizations
                torch.backends.mps.enable_fallback()
            
            # Set memory allocation strategy
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
            
            logger.info("M4 inference optimizations applied")
            
        except Exception as e:
            logger.error(f"Error applying optimizations: {e}")
    
    def shutdown(self):
        """Clean shutdown of model manager"""
        self.memory_monitor.stop_monitoring()
        
        # Unload all models
        for model_name in list(self.loaded_models.keys()):
            self.unload_model(model_name)
        
        # Close cache
        self.cache.close()
        
        logger.info("ModelManager shutdown complete")


class BatchProcessor:
    """Optimized batch processing for M4 Mac"""
    
    def __init__(self, batch_size: int = 4, max_memory_mb: int = 2048):
        self.batch_size = batch_size
        self.max_memory_mb = max_memory_mb
        self.memory_monitor = MemoryMonitor()
    
    async def process_batch(self, items: List[Any], processor_func, **kwargs) -> List[Any]:
        """Process items in optimized batches"""
        results = []
        
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            
            # Check memory before processing
            stats = self.memory_monitor.get_memory_stats()
            if stats.ram_percent > 0.85:
                # Reduce batch size temporarily
                batch = batch[:max(1, len(batch) // 2)]
                logger.warning(f"Reduced batch size due to memory pressure: {len(batch)}")
            
            try:
                batch_results = await processor_func(batch, **kwargs)
                results.extend(batch_results)
                
            except Exception as e:
                logger.error(f"Error processing batch {i//self.batch_size}: {e}")
                # Process items individually as fallback
                for item in batch:
                    try:
                        result = await processor_func([item], **kwargs)
                        results.extend(result)
                    except Exception as item_error:
                        logger.error(f"Error processing individual item: {item_error}")
                        results.append(None)
        
        return results
    
    def adaptive_batch_size(self, base_size: int = 4) -> int:
        """Calculate adaptive batch size based on memory"""
        stats = self.memory_monitor.get_memory_stats()
        
        if stats.ram_percent > 0.8:
            return max(1, base_size // 2)
        elif stats.ram_percent < 0.6:
            return min(8, base_size * 2)
        else:
            return base_size


class StreamingProcessor:
    """Memory-efficient streaming processor for large documents"""
    
    def __init__(self, chunk_size: int = 2000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.memory_monitor = MemoryMonitor()
    
    def stream_process(self, content: str, processor_func):
        """Stream process content in chunks"""
        chunks = self._create_chunks(content)
        
        for i, chunk in enumerate(chunks):
            # Check memory
            stats = self.memory_monitor.get_memory_stats()
            if stats.ram_percent > 0.85:
                logger.warning("High memory usage during streaming")
                gc.collect()
            
            try:
                yield processor_func(chunk, chunk_id=i)
            except Exception as e:
                logger.error(f"Error processing chunk {i}: {e}")
                yield None
    
    def _create_chunks(self, content: str) -> List[str]:
        """Create overlapping chunks from content"""
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + self.chunk_size
            chunk = content[start:end]
            
            # Find good breaking point
            if end < len(content):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > start + self.chunk_size // 2:
                    chunk = content[start:break_point + 1]
                    end = break_point + 1
            
            chunks.append(chunk.strip())
            start = end - self.overlap
            
            if start >= len(content):
                break
        
        return chunks


def setup_m4_optimization():
    """Setup system-wide M4 optimizations"""
    try:
        # Set environment variables for optimal performance
        os.environ['OMP_NUM_THREADS'] = '8'  # M4 performance cores
        os.environ['MKL_NUM_THREADS'] = '8'
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        
        # PyTorch optimizations
        torch.set_num_threads(8)
        
        if torch.backends.mps.is_available():
            torch.backends.mps.enable_fallback()
            logger.info("MPS acceleration enabled")
        
        # Memory optimizations
        gc.set_threshold(700, 10, 10)  # Aggressive garbage collection
        
        logger.info("M4 Mac optimizations configured successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error setting up M4 optimizations: {e}")
        return False


def get_system_info() -> Dict[str, Any]:
    """Get system information for optimization"""
    info = {
        'cpu_count': psutil.cpu_count(),
        'memory_gb': psutil.virtual_memory().total / (1024**3),
        'mps_available': torch.backends.mps.is_available() if torch.is_available() else False,
        'mlx_available': MLX_AVAILABLE,
        'platform': os.uname().machine if hasattr(os, 'uname') else 'unknown'
    }
    
    return info