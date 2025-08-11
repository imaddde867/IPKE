#!/usr/bin/env python3
"""
SIMPLIFIED Optimized Document Processor - Core Structure Only
For testing and validation without external dependencies
"""

import asyncio
import logging
import time
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import threading
import os
import io
import tempfile
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """Optimized processing result"""
    document_id: str
    document_type: str
    processing_time: float
    entities_extracted: int
    confidence_score: float
    performance_metrics: Dict[str, Any]
    entities: List[Dict[str, Any]]
    processing_method: str
    cache_hit: bool
    optimization_level: str
    content_summary: str
    file_size: int
    format_detected: str

class SimpleLLMEngine:
    """Simplified LLM engine for testing"""
    
    def __init__(self):
        self.cache = {}
    
    def process_document_optimized(self, content: str, document_type: str) -> Dict[str, Any]:
        """Simulate LLM processing"""
        # Simulate processing time
        time.sleep(0.1)
        
        # Return mock results
        return {
            "entities": [
                {"content": "Sample entity", "entity_type": "concept", "confidence": 0.8}
            ],
            "confidence": 0.8,
            "method": "simple_llm"
        }
    
    def cleanup(self):
        """Cleanup resources"""
        self.cache.clear()

class SimpleExtractionEngine:
    """Simplified extraction engine for testing"""
    
    def __init__(self):
        self.pattern_cache = {}
        self.entity_cache = {}
    
    def extract_comprehensive_knowledge(self, content: str, document_type: str) -> List[Dict[str, Any]]:
        """Simulate entity extraction"""
        # Simulate processing time
        time.sleep(0.1)
        
        # Return mock entities
        return [
            {"content": "Extracted entity", "entity_type": "concept", "confidence": 0.7}
        ]
    
    def cleanup(self):
        """Cleanup resources"""
        self.pattern_cache.clear()
        self.entity_cache.clear()

class OptimizedDocumentProcessorSimple:
    """SIMPLIFIED optimized document processor for testing"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Performance optimizations
        self.max_workers = 4  # Optimize for M4 chip
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.processing_cache = {}
        self.cache_lock = threading.Lock()
        
        # Initialize engines (lazy loading)
        self.llm_engine = None
        self.extraction_engine = None
        self.engines_initialized = False
        
        # Performance monitoring
        self.processing_stats = {
            "total_documents": 0,
            "total_processing_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "average_processing_time": 0.0,
            "performance_target_met": 0,
            "performance_target_missed": 0,
            "format_processing_times": {},
            "error_count": 0
        }
        
        # Performance thresholds
        self.target_processing_time = 120.0  # 2 minutes target
        self.performance_warning_threshold = 90.0  # 1.5 minutes warning
        
        # Supported formats (simplified)
        self.supported_formats = {
            'text': ['.txt', '.md'],
            'image': ['.jpg', '.png'],
            'document': ['.pdf', '.docx']
        }
        
        print("🚀 SIMPLIFIED Optimized Document Processor initialized")
        print(f"🎯 Performance Target: {self.target_processing_time} seconds per document")
        print(f"⚡ Max Workers: {self.max_workers} (optimized for M4)")
    
    def _initialize_engines(self):
        """Lazy initialize processing engines"""
        if self.engines_initialized:
            return
        
        try:
            # Initialize simplified engines
            self.llm_engine = SimpleLLMEngine()
            self.extraction_engine = SimpleExtractionEngine()
            self.engines_initialized = True
            
            print("✅ Simplified engines initialized")
            
        except Exception as e:
            print(f"⚠️ Engine initialization failed: {e}")
            logger.error(f"Engine initialization failed: {e}")
    
    async def process_document_async(self, file_path: str, document_id: str = None) -> ProcessingResult:
        """OPTIMIZED async document processing"""
        start_time = time.time()
        
        # Generate document ID if not provided
        if not document_id:
            file_path_obj = Path(file_path)
            content_hash = hashlib.md5(f"{file_path}_{start_time}".encode()).hexdigest()
            document_id = f"doc_{file_path_obj.stem}_{content_hash[:8]}"
        
        # Check cache first
        cache_key = self._generate_cache_key(file_path)
        cached_result = self._get_cached_result(cache_key)
        
        if cached_result:
            self.processing_stats["cache_hits"] += 1
            return cached_result
        
        self.processing_stats["cache_misses"] += 1
        
        try:
            # Get file info
            file_path_obj = Path(file_path)
            file_size = file_path_obj.stat().st_size if file_path_obj.exists() else 0
            file_extension = file_path_obj.suffix.lower()
            format_detected = self._get_file_type(file_extension)
            
            # Extract content (simplified)
            content = await self._extract_content_async(file_path, format_detected)
            
            # Initialize engines if needed
            self._initialize_engines()
            
            # Parallel processing pipeline
            processing_tasks = [
                self._process_with_llm_async(content, format_detected),
                self._extract_entities_async(content, format_detected),
                self._validate_and_enhance_async(content, format_detected)
            ]
            
            # Execute all tasks in parallel
            try:
                results = await asyncio.gather(*processing_tasks, return_exceptions=True)
                
                # Process results
                llm_result = results[0] if not isinstance(results[0], Exception) else None
                extraction_result = results[1] if not isinstance(results[1], Exception) else None
                validation_result = results[2] if not isinstance(results[2], Exception) else None
                
            except Exception as e:
                logger.error(f"Parallel processing failed: {e}")
                # Fallback to sequential processing
                llm_result = await self._process_with_llm_async(content, format_detected)
                extraction_result = await self._extract_entities_async(content, format_detected)
                validation_result = await self._validate_and_enhance_async(content, format_detected)
            
            # Merge and finalize results
            final_result = self._merge_processing_results(
                llm_result, extraction_result, validation_result, 
                content, format_detected, document_id, file_size
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            final_result.processing_time = processing_time
            
            # Update performance stats
            self._update_performance_stats(processing_time, format_detected)
            
            # Cache the result
            self._cache_result(cache_key, final_result)
            
            # Performance monitoring
            self._monitor_performance(processing_time, document_id)
            
            return final_result
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            self.processing_stats["error_count"] += 1
            
            # Return error result
            return ProcessingResult(
                document_id=document_id,
                document_type="error",
                processing_time=time.time() - start_time,
                entities_extracted=0,
                confidence_score=0.0,
                performance_metrics={"error": str(e)},
                entities=[],
                processing_method="error",
                cache_hit=False,
                optimization_level="error",
                content_summary=f"Processing failed: {str(e)}",
                file_size=0,
                format_detected="unknown"
            )
    
    def process_document_sync(self, file_path: str, document_id: str = None) -> ProcessingResult:
        """Synchronous wrapper for async processing"""
        try:
            return asyncio.run(self.process_document_async(file_path, document_id))
        except RuntimeError:
            # If already in event loop, create new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self.process_document_async(file_path, document_id)
                )
            finally:
                loop.close()
    
    async def _extract_content_async(self, file_path: str, format_type: str) -> str:
        """Async content extraction based on format"""
        try:
            # Use ThreadPoolExecutor for I/O-bound operations
            loop = asyncio.get_event_loop()
            
            if format_type == "text":
                return await loop.run_in_executor(
                    self.executor, self._extract_text_content, file_path
                )
            elif format_type == "image":
                return await loop.run_in_executor(
                    self.executor, self._extract_image_content, file_path
                )
            else:
                return await loop.run_in_executor(
                    self.executor, self._extract_generic_content, file_path
                )
                
        except Exception as e:
            logger.error(f"Content extraction failed: {e}")
            return f"Content extraction failed: {str(e)}"
    
    def _extract_text_content(self, file_path: str) -> str:
        """Extract text content from text files"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                return file.read()
        except Exception as e:
            return f"Text extraction failed: {str(e)}"
    
    def _extract_image_content(self, file_path: str) -> str:
        """Extract content from images (simplified)"""
        return f"Image file: {Path(file_path).name} (OCR simulation)"
    
    def _extract_generic_content(self, file_path: str) -> str:
        """Generic content extraction fallback"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                return file.read()
        except Exception:
            return f"Generic content extraction for {Path(file_path).name}"
    
    async def _process_with_llm_async(self, content: str, document_type: str) -> Dict[str, Any]:
        """Async LLM processing"""
        if not self.llm_engine:
            return {"entities": [], "confidence": 0.0, "method": "llm_unavailable"}
        
        try:
            # Use ThreadPoolExecutor for CPU-bound LLM processing
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self.llm_engine.process_document_optimized,
                content,
                document_type
            )
            return result
        except Exception as e:
            logger.error(f"LLM processing failed: {e}")
            return {"entities": [], "confidence": 0.0, "method": "llm_failed"}
    
    async def _extract_entities_async(self, content: str, document_type: str) -> Dict[str, Any]:
        """Async entity extraction"""
        if not self.extraction_engine:
            return {"entities": [], "confidence": 0.0, "method": "extraction_unavailable"}
        
        try:
            # Use ThreadPoolExecutor for CPU-bound extraction
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self.extraction_engine.extract_comprehensive_knowledge,
                content,
                document_type
            )
            
            # Convert to dict format
            entities_dict = []
            for entity in result:
                entity_dict = asdict(entity) if hasattr(entity, '__dataclass_fields__') else entity
                entities_dict.append(entity_dict)
            
            return {
                "entities": entities_dict,
                "confidence": 0.7,  # Simplified confidence
                "method": "enhanced_extraction"
            }
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return {"entities": [], "confidence": 0.0, "method": "extraction_failed"}
    
    async def _validate_and_enhance_async(self, content: str, document_type: str) -> Dict[str, Any]:
        """Async validation and enhancement"""
        try:
            # Fast validation and enhancement
            validation_result = {
                "entities": [],
                "confidence": 0.0,
                "method": "fast_validation"
            }
            
            # Basic content validation
            if len(content) < 10:
                validation_result["confidence"] = 0.1
            elif len(content) < 100:
                validation_result["confidence"] = 0.3
            elif len(content) < 1000:
                validation_result["confidence"] = 0.6
            else:
                validation_result["confidence"] = 0.8
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {"entities": [], "confidence": 0.0, "method": "validation_failed"}
    
    def _merge_processing_results(self, llm_result: Dict[str, Any], 
                                extraction_result: Dict[str, Any],
                                validation_result: Dict[str, Any],
                                content: str, document_type: str, 
                                document_id: str, file_size: int) -> ProcessingResult:
        """Merge results from different processing methods"""
        
        # Combine all entities
        all_entities = []
        all_entities.extend(llm_result.get("entities", []))
        all_entities.extend(extraction_result.get("entities", []))
        
        # Remove duplicates (simple deduplication)
        unique_entities = self._deduplicate_entities(all_entities)
        
        # Calculate overall confidence
        confidences = [
            llm_result.get("confidence", 0.0),
            extraction_result.get("confidence", 0.0),
            validation_result.get("confidence", 0.0)
        ]
        overall_confidence = sum(confidences) / len(confidences)
        
        # Determine processing method
        methods = [
            llm_result.get("method", "unknown"),
            extraction_result.get("method", "unknown"),
            validation_result.get("method", "unknown")
        ]
        primary_method = methods[0] if methods else "unknown"
        
        # Performance metrics
        performance_metrics = {
            "llm_confidence": llm_result.get("confidence", 0.0),
            "extraction_confidence": extraction_result.get("confidence", 0.0),
            "validation_confidence": validation_result.get("confidence", 0.0),
            "total_entities": len(unique_entities),
            "processing_methods": methods,
            "optimization_level": "high"
        }
        
        # Generate content summary
        content_summary = self._generate_content_summary(content, unique_entities)
        
        return ProcessingResult(
            document_id=document_id,
            document_type=document_type,
            processing_time=0.0,  # Will be set later
            entities_extracted=len(unique_entities),
            confidence_score=overall_confidence,
            performance_metrics=performance_metrics,
            entities=unique_entities,
            processing_method=primary_method,
            cache_hit=False,
            optimization_level="high",
            content_summary=content_summary,
            file_size=file_size,
            format_detected=document_type
        )
    
    def _deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fast entity deduplication"""
        if not entities:
            return entities
        
        seen = set()
        unique_entities = []
        
        for entity in entities:
            # Create a simple hash for deduplication
            content = entity.get("content", "")
            entity_type = entity.get("entity_type", "")
            category = entity.get("category", "")
            
            entity_hash = f"{content[:50]}_{entity_type}_{category}"
            
            if entity_hash not in seen:
                seen.add(entity_hash)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _generate_content_summary(self, content: str, entities: List[Dict[str, Any]]) -> str:
        """Generate fast content summary"""
        if not content:
            return "No content available"
        
        # Simple summary based on content length and entities
        word_count = len(content.split())
        entity_count = len(entities)
        
        if word_count < 100:
            summary = f"Short document ({word_count} words)"
        elif word_count < 1000:
            summary = f"Medium document ({word_count} words)"
        else:
            summary = f"Long document ({word_count} words)"
        
        if entity_count > 0:
            summary += f" with {entity_count} extracted entities"
        
        return summary
    
    def _get_file_type(self, extension: str) -> str:
        """Get file type from extension"""
        for file_type, extensions in self.supported_formats.items():
            if extension in extensions:
                return file_type
        return "unknown"
    
    def _generate_cache_key(self, file_path: str) -> str:
        """Generate cache key for file"""
        file_path_obj = Path(file_path)
        try:
            file_hash = hashlib.md5(f"{file_path}_{file_path_obj.stat().st_mtime}".encode()).hexdigest()
        except:
            file_hash = hashlib.md5(file_path.encode()).hexdigest()
        return f"{file_path_obj.suffix}_{file_hash}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[ProcessingResult]:
        """Get cached processing result"""
        with self.cache_lock:
            return self.processing_cache.get(cache_key)
    
    def _cache_result(self, cache_key: str, result: ProcessingResult):
        """Cache processing result"""
        with self.cache_lock:
            # Limit cache size for memory efficiency
            if len(self.processing_cache) > 100:
                # Remove oldest entries
                oldest_keys = list(self.processing_cache.keys())[:20]
                for key in oldest_keys:
                    del self.processing_cache[key]
            
            self.processing_cache[cache_key] = result
    
    def _update_performance_stats(self, processing_time: float, format_type: str):
        """Update performance statistics"""
        self.processing_stats["total_documents"] += 1
        self.processing_stats["total_processing_time"] += processing_time
        
        # Calculate running average
        total_docs = self.processing_stats["total_documents"]
        total_time = self.processing_stats["total_processing_time"]
        self.processing_stats["average_processing_time"] = total_time / total_docs
        
        # Track format-specific performance
        if format_type not in self.processing_stats["format_processing_times"]:
            self.processing_stats["format_processing_times"][format_type] = []
        
        self.processing_stats["format_processing_times"][format_type].append(processing_time)
        
        # Keep only last 50 times per format for memory efficiency
        if len(self.processing_stats["format_processing_times"][format_type]) > 50:
            self.processing_stats["format_processing_times"][format_type] = \
                self.processing_stats["format_processing_times"][format_type][-50:]
        
        # Track target performance
        if processing_time <= self.target_processing_time:
            self.processing_stats["performance_target_met"] += 1
        else:
            self.processing_stats["performance_target_missed"] += 1
    
    def _monitor_performance(self, processing_time: float, document_id: str):
        """Monitor and report performance"""
        if processing_time > self.target_processing_time:
            print(f"⚠️ PERFORMANCE WARNING: Document {document_id} took {processing_time:.2f}s "
                  f"(target: {self.target_processing_time}s)")
        elif processing_time > self.performance_warning_threshold:
            print(f"⚠️ PERFORMANCE WARNING: Document {document_id} took {processing_time:.2f}s "
                  f"(approaching target limit)")
        else:
            print(f"✅ Document {document_id} processed in {processing_time:.2f}s "
                  f"(target: {self.target_processing_time}s)")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        total_docs = self.processing_stats["total_documents"]
        
        if total_docs == 0:
            return {
                "status": "no_documents_processed",
                "performance_optimized": True,
                "target_met": False
            }
        
        cache_hit_rate = (self.processing_stats["cache_hits"] / 
                         (self.processing_stats["cache_hits"] + self.processing_stats["cache_misses"]))
        
        target_success_rate = (self.processing_stats["performance_target_met"] / total_docs)
        
        # Format-specific performance
        format_performance = {}
        for format_type, times in self.processing_stats["format_processing_times"].items():
            if times:
                format_performance[format_type] = {
                    "average_time": sum(times) / len(times),
                    "total_processed": len(times),
                    "target_met_rate": sum(1 for t in times if t <= self.target_processing_time) / len(times)
                }
        
        return {
            "total_documents_processed": total_docs,
            "average_processing_time": self.processing_stats["average_processing_time"],
            "cache_hit_rate": cache_hit_rate,
            "performance_target_success_rate": target_success_rate,
            "performance_target_met_count": self.processing_stats["performance_target_met"],
            "performance_target_missed_count": self.processing_stats["performance_target_missed"],
            "error_count": self.processing_stats["error_count"],
            "format_performance": format_performance,
            "performance_optimized": True,
            "target_met": target_success_rate >= 0.8,  # 80% success rate
            "current_performance_vs_target": {
                "target_time": self.target_processing_time,
                "current_average": self.processing_stats["average_processing_time"],
                "improvement_factor": (10.0 * 60.0) / self.processing_stats["average_processing_time"]  # vs 10 minutes
            }
        }
    
    def optimize_for_m4(self):
        """Apply M4-specific optimizations"""
        print("🔧 Applying M4-specific optimizations...")
        
        # Adjust thread pool for M4 efficiency cores
        self.max_workers = 6  # M4 has 6 efficiency cores
        if self.executor:
            self.executor.shutdown(wait=True)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Memory optimization
        self.target_processing_time = 120.0  # 2 minutes
        self.performance_warning_threshold = 90.0  # 1.5 minutes
        
        print(f"✅ M4 optimizations applied: {self.max_workers} workers, "
              f"{self.target_processing_time}s target")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.executor:
            self.executor.shutdown(wait=True)
        
        if self.llm_engine:
            self.llm_engine.cleanup()
        
        if self.extraction_engine:
            self.extraction_engine.cleanup()
        
        # Clear cache
        with self.cache_lock:
            self.processing_cache.clear()
        
        print("🧹 Simplified Optimized Document Processor cleaned up")

# Backward compatibility
DocumentProcessor = OptimizedDocumentProcessorSimple
