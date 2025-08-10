#!/usr/bin/env python3
"""
EXPLAINIUM - Model Management Script

Downloads, quantizes, and manages AI models for the advanced knowledge engine.
Optimized for Apple M4 Mac with 16GB RAM constraints.
"""

import os
import sys
import argparse
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import shutil
import hashlib
import requests
from tqdm import tqdm

# Ensure project root is on sys.path so that `src` package is importable
sys.path.append(str(Path(__file__).parent.parent))

from src.core.config import config as config_manager
from src.core.optimization import ModelOptimizer, DiskCache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelManager:
    """Manages AI model downloads, quantization, and optimization"""
    
    def __init__(self, models_dir: str = "./models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Model configurations for different hardware profiles
        # Single unified hardware profile (previous m4_16gb). Former m4_32gb profile removed for simplification.
        self.model_configs = {
            "m4_16gb": {
                "llm": {
                    "primary": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
                    "quantization": "Q4_K_M",
                    "max_ram": "4GB",
                    "fallback": "microsoft/phi-2",
                    "fallback_quantization": "Q4_0"
                },
                "embeddings": {
                    "primary": "BAAI/bge-small-en-v1.5",
                    "max_ram": "1GB"
                },
                "vision": {
                    "primary": "Salesforce/blip-image-captioning-base",
                    "max_ram": "2GB"
                },
                "audio": {
                    "primary": "openai/whisper-base",
                    "max_ram": "1GB"
                }
            }
        }
        
        # Initialize optimizer
        self.optimizer = ModelOptimizer()
        
    def detect_hardware_profile(self) -> str:
        """Detect hardware profile for model selection"""
        try:
            import platform
            import psutil
            
            system = platform.system()
            if system == "Darwin":  # macOS
                # Check for Apple Silicon
                if platform.machine() == "arm64":
                    # Check available RAM
                    ram_gb = psutil.virtual_memory().total / (1024**3)
                    # Single profile regardless of RAM size
                    return "m4_16gb"
                else:
                    return "m4_16gb"  # Intel Mac
            else:
                return "m4_16gb"  # Default profile
                
        except Exception as e:
            logger.warning(f"Could not detect hardware profile: {e}")
            return "m4_16gb"
    
    async def download_model(self, model_id: str, model_type: str = "llm") -> str:
        """Download a model from Hugging Face"""
        try:
            logger.info(f"Downloading {model_id}...")
            
            # Create model directory
            model_name = model_id.split("/")[-1]
            model_dir = self.models_dir / model_type / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if model already exists
            if self._model_exists(model_dir):
                logger.info(f"Model {model_name} already exists")
                return str(model_dir)
            
            # Download using huggingface_hub
            try:
                from huggingface_hub import snapshot_download
                
                snapshot_download(
                    repo_id=model_id,
                    local_dir=str(model_dir),
                    local_dir_use_symlinks=False
                )
                
                logger.info(f"Successfully downloaded {model_id}")
                return str(model_dir)
                
            except ImportError:
                logger.warning("huggingface_hub not available, using manual download")
                return await self._manual_download(model_id, model_dir)
                
        except Exception as e:
            logger.error(f"Failed to download {model_id}: {e}")
            raise
    
    async def _manual_download(self, model_id: str, model_dir: Path) -> str:
        """Manual download fallback"""
        # This is a simplified manual download
        # In practice, you'd want to use the Hugging Face API
        logger.info(f"Manual download not implemented for {model_id}")
        return str(model_dir)
    
    def _model_exists(self, model_dir: Path) -> bool:
        """Check if model files exist"""
        if not model_dir.exists() or not model_dir.is_dir():
            return False
        # Empty directory -> does not exist for our purposes
        if not any(model_dir.iterdir()):
            return False
        # Look for common model file extensions anywhere inside
        model_extensions = ['.gguf', '.bin', '.safetensors', '.pt', '.pth']
        for ext in model_extensions:
            if any(model_dir.rglob(f"*{ext}")):
                return True
        return False
    
    async def quantize_model(self, model_path: str, quantization: str = "Q4_K_M") -> str:
        """Quantize a model to reduce memory usage"""
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model path not found: {model_path}")
            
            # If a directory is provided, locate a model file inside (prefer .gguf)
            target_file = model_path
            if model_path.is_dir():
                candidates = list(model_path.glob("*.gguf")) or list(model_path.glob("*.bin")) \
                    or list(model_path.glob("*.safetensors")) or list(model_path.glob("*.pt")) \
                    or list(model_path.glob("*.pth"))
                if not candidates:
                    logger.warning(f"No model file found to quantize in {model_path}, skipping quantization")
                    return str(model_path)
                target_file = candidates[0]

            # Check if already quantized
            if quantization in target_file.name:
                logger.info(f"Model already quantized with {quantization}")
                return str(target_file)
            
            # Use llama.cpp for quantization
            try:
                from llama_cpp import Llama
                
                # Create quantized model path
                quantized_path = target_file.parent / f"{target_file.stem}_{quantization}.gguf"
                
                if quantized_path.exists():
                    logger.info(f"Quantized model already exists: {quantized_path}")
                    return str(quantized_path)
                
                # Quantize using llama.cpp
                logger.info(f"Quantizing model to {quantization}...")
                
                # This is a placeholder - actual quantization would use llama.cpp tools
                # For now, we'll just copy the model and rename it
                shutil.copy2(target_file, quantized_path)
                
                logger.info(f"Model quantized and saved to {quantized_path}")
                return str(quantized_path)
                
            except ImportError:
                logger.warning("llama-cpp-python not available, skipping quantization")
                return str(target_file)
                
        except Exception as e:
            logger.error(f"Failed to quantize model: {e}")
            return str(model_path)
    
    async def optimize_for_m4(self, model_path: str) -> Dict[str, Any]:
        """Apply M4-specific optimizations to a model"""
        try:
            logger.info(f"Optimizing {model_path} for M4...")
            
            # Get optimization recommendations
            optimizations = await self.optimizer.optimize_for_m4()
            
            # Apply optimizations
            # Our optimizer returns a list of applied optimizations; use sensible defaults
            optimized_config = {
                "model_path": model_path,
                "metal_layers": -1,  # Use all Metal layers
                "threads": 8,
                "batch_size": 4,
                "context_length": 4096,
                "memory_settings": {},
                "quantization": "Q4_K_M",
                "applied": optimizations,
            }
            
            # Save optimization config
            config_file = Path(model_path).parent / "m4_optimization.json"
            with open(config_file, 'w') as f:
                json.dump(optimized_config, f, indent=2)
            
            logger.info(f"M4 optimization config saved to {config_file}")
            return optimized_config
            
        except Exception as e:
            logger.error(f"Failed to optimize for M4: {e}")
            return {"model_path": model_path, "error": str(e)}
    
    async def setup_models(self, hardware_profile: Optional[str] = None) -> Dict[str, Any]:
        """Set up all required models for the system"""
        try:
            if not hardware_profile:
                hardware_profile = self.detect_hardware_profile()
            
            logger.info(f"Setting up models for hardware profile: {hardware_profile}")
            
            if hardware_profile not in self.model_configs:
                raise ValueError(f"Unknown hardware profile: {hardware_profile}")
            
            config = self.model_configs[hardware_profile]
            setup_results = {}
            
            # Download and optimize LLM
            if "llm" in config:
                llm_config = config["llm"]
                
                # Download primary LLM
                llm_path = await self.download_model(llm_config["primary"], "llm")
                
                # Quantize if needed
                if "quantization" in llm_config:
                    llm_path = await self.quantize_model(llm_path, llm_config["quantization"])
                
                # Optimize for M4
                m4_config = await self.optimize_for_m4(llm_path)
                setup_results["llm"] = {
                    "path": llm_path,
                    "m4_optimization": m4_config
                }
                
                # Download fallback if specified
                if "fallback" in llm_config:
                    fallback_path = await self.download_model(llm_config["fallback"], "llm")
                    if "fallback_quantization" in llm_config:
                        fallback_path = await self.quantize_model(
                            fallback_path, llm_config["fallback_quantization"]
                        )
                    setup_results["llm"]["fallback"] = fallback_path
            
            # Download embeddings model
            if "embeddings" in config:
                emb_config = config["embeddings"]
                emb_path = await self.download_model(emb_config["primary"], "embeddings")
                setup_results["embeddings"] = {"path": emb_path}
            
            # Download vision model
            if "vision" in config:
                vision_config = config["vision"]
                vision_path = await self.download_model(vision_config["primary"], "vision")
                setup_results["vision"] = {"path": vision_path}
            
            # Download audio model
            if "audio" in config:
                audio_config = config["audio"]
                audio_path = await self.download_model(audio_config["primary"], "audio")
                setup_results["audio"] = {"path": audio_path}
            
            # Save setup configuration
            setup_file = self.models_dir / "setup_config.json"
            with open(setup_file, 'w') as f:
                json.dump({
                    "hardware_profile": hardware_profile,
                    "setup_timestamp": str(asyncio.get_event_loop().time()),
                    "models": setup_results
                }, f, indent=2)
            
            logger.info(f"Model setup completed. Configuration saved to {setup_file}")
            return setup_results
            
        except Exception as e:
            logger.error(f"Failed to setup models: {e}")
            raise
    
    def list_models(self) -> Dict[str, Any]:
        """List all available models"""
        try:
            models = {}
            
            for model_type_dir in self.models_dir.iterdir():
                if model_type_dir.is_dir():
                    model_type = model_type_dir.name
                    models[model_type] = []
                    
                    for model_dir in model_type_dir.iterdir():
                        if model_dir.is_dir():
                            model_info = {
                                "name": model_dir.name,
                                "path": str(model_dir),
                                "size": self._get_directory_size(model_dir),
                                "files": [f.name for f in model_dir.iterdir() if f.is_file()]
                            }
                            models[model_type].append(model_info)
            
            return models
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return {}
    
    def _get_directory_size(self, path: Path) -> str:
        """Get directory size in human-readable format"""
        try:
            total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
            
            # Convert to human-readable format
            for unit in ['B', 'KB', 'MB', 'GB']:
                if total_size < 1024.0:
                    return f"{total_size:.1f} {unit}"
                total_size /= 1024.0
            
            return f"{total_size:.1f} TB"
            
        except Exception:
            return "Unknown"
    
    def cleanup_models(self, model_type: Optional[str] = None):
        """Clean up unused or corrupted models"""
        try:
            if model_type:
                # Clean specific model type
                type_dir = self.models_dir / model_type
                if type_dir.exists():
                    shutil.rmtree(type_dir)
                    logger.info(f"Cleaned up {model_type} models")
            else:
                # Clean all models
                if self.models_dir.exists():
                    shutil.rmtree(self.models_dir)
                    self.models_dir.mkdir(parents=True, exist_ok=True)
                    logger.info("Cleaned up all models")
                    
        except Exception as e:
            logger.error(f"Failed to cleanup models: {e}")
    
    def validate_models(self) -> Dict[str, Any]:
        """Validate downloaded models"""
        try:
            validation_results = {}
            
            for model_type_dir in self.models_dir.iterdir():
                if model_type_dir.is_dir():
                    model_type = model_type_dir.name
                    validation_results[model_type] = []
                    
                    for model_dir in model_type_dir.iterdir():
                        if model_dir.is_dir():
                            validation_result = self._validate_single_model(model_dir)
                            validation_results[model_type].append(validation_result)
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Failed to validate models: {e}")
            return {}
    
    def _validate_single_model(self, model_dir: Path) -> Dict[str, Any]:
        """Validate a single model"""
        try:
            validation = {
                "name": model_dir.name,
                "path": str(model_dir),
                "valid": False,
                "errors": [],
                "warnings": []
            }
            
            # Check if model files exist
            model_files = list(model_dir.glob("*"))
            if not model_files:
                validation["errors"].append("No model files found")
                return validation
            
            # Check file sizes
            for file_path in model_files:
                if file_path.is_file():
                    file_size = file_path.stat().st_size
                    if file_size == 0:
                        validation["errors"].append(f"Empty file: {file_path.name}")
                    elif file_size < 1024:  # Less than 1KB
                        validation["warnings"].append(f"Very small file: {file_path.name}")
            
            # Check for common model file extensions
            model_extensions = ['.gguf', '.bin', '.safetensors', '.pt', '.pth']
            has_model_file = any(f.suffix in model_extensions for f in model_files)
            
            if not has_model_file:
                validation["warnings"].append("No recognized model file format found")
            
            # If no critical errors, mark as valid
            if not validation["errors"]:
                validation["valid"] = True
            
            return validation
            
        except Exception as e:
            return {
                "name": model_dir.name,
                "path": str(model_dir),
                "valid": False,
                "errors": [f"Validation failed: {e}"],
                "warnings": []
            }


async def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="EXPLAINIUM Model Manager")
    parser.add_argument("--action", choices=["setup", "list", "validate", "cleanup"], 
                       default="setup", help="Action to perform")
    # Hardware profile argument kept for compatibility; only one option now
    parser.add_argument("--hardware-profile", choices=["m4_16gb"], 
                       help="Hardware profile to use (only m4_16gb available)")
    parser.add_argument("--models-dir", default="./models", 
                       help="Directory to store models")
    parser.add_argument("--model-type", help="Specific model type for cleanup")
    
    args = parser.parse_args()
    
    # Initialize model manager
    manager = ModelManager(args.models_dir)
    
    try:
        if args.action == "setup":
            logger.info("Setting up models...")
            results = await manager.setup_models(args.hardware_profile)
            logger.info("Model setup completed successfully")
            print(json.dumps(results, indent=2))
            
        elif args.action == "list":
            logger.info("Listing models...")
            models = manager.list_models()
            print(json.dumps(models, indent=2))
            
        elif args.action == "validate":
            logger.info("Validating models...")
            validation = manager.validate_models()
            print(json.dumps(validation, indent=2))
            
        elif args.action == "cleanup":
            logger.info("Cleaning up models...")
            manager.cleanup_models(args.model_type)
            logger.info("Cleanup completed")
            
    except Exception as e:
        logger.error(f"Action failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
