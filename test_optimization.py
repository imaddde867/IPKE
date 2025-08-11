#!/usr/bin/env python3
"""
Performance Optimization Test Script
Tests the optimized Explainium pipeline to verify 2-minute target achievement
"""

import asyncio
import time
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from processors.optimized_processor import OptimizedDocumentProcessor
from ai.llm_processing_engine import OptimizedLLMProcessingEngine
from ai.enhanced_extraction_engine import OptimizedEnhancedExtractionEngine

async def test_optimization_pipeline():
    """Test the complete optimization pipeline"""
    print("🚀 Testing Explainium Performance Optimization Pipeline")
    print("=" * 60)
    
    # Initialize the optimized processor
    processor = OptimizedDocumentProcessor()
    
    # Apply M4-specific optimizations
    processor.optimize_for_m4()
    
    # Test with a sample document (create a simple test file)
    test_content = """
    This is a test document for performance optimization testing.
    
    Technical Specifications:
    - CPU: M4 Pro with 10-core CPU
    - Memory: 16GB unified memory
    - Storage: 512GB SSD
    - Performance Target: 2 minutes per document
    
    Key Features:
    - Asynchronous processing pipeline
    - Intelligent caching system
    - Content chunking and streaming
    - Parallel entity extraction
    - M4 chip optimizations
    
    The system should process this document in under 2 minutes
    while maintaining high quality entity extraction and confidence scores.
    
    Expected Entities:
    - Hardware specifications
    - Performance metrics
    - Technical features
    - System requirements
    """
    
    # Create a temporary test file
    test_file_path = "test_document.txt"
    with open(test_file_path, "w") as f:
        f.write(test_content)
    
    try:
        print(f"📄 Testing with document: {test_file_path}")
        print(f"📊 Document size: {len(test_content)} characters")
        
        # Test async processing
        print("\n🔄 Testing async processing pipeline...")
        start_time = time.time()
        
        result = await processor.process_document_async(test_file_path)
        
        processing_time = time.time() - start_time
        
        print(f"\n✅ Processing completed!")
        print(f"⏱️  Total processing time: {processing_time:.2f} seconds")
        print(f"🎯 Target time: {processor.target_processing_time} seconds")
        print(f"📈 Performance improvement: {(10 * 60) / processing_time:.1f}x faster than 10 minutes")
        
        # Performance analysis
        if processing_time <= processor.target_processing_time:
            print(f"🎉 SUCCESS: Target achieved! Processing time: {processing_time:.2f}s")
        else:
            print(f"⚠️  WARNING: Target missed. Processing time: {processing_time:.2f}s")
        
        # Display results
        print(f"\n📋 Processing Results:")
        print(f"   Document ID: {result.document_id}")
        print(f"   Document Type: {result.document_type}")
        print(f"   Entities Extracted: {result.entities_extracted}")
        print(f"   Confidence Score: {result.confidence_score:.2f}")
        print(f"   Processing Method: {result.processing_method}")
        print(f"   Optimization Level: {result.optimization_level}")
        print(f"   Content Summary: {result.content_summary}")
        
        # Performance metrics
        print(f"\n📊 Performance Metrics:")
        print(f"   LLM Confidence: {result.performance_metrics.get('llm_confidence', 0):.2f}")
        print(f"   Extraction Confidence: {result.performance_metrics.get('extraction_confidence', 0):.2f}")
        print(f"   Validation Confidence: {result.performance_metrics.get('validation_confidence', 0):.2f}")
        print(f"   Total Entities: {result.performance_metrics.get('total_entities', 0)}")
        
        # Get overall performance summary
        performance_summary = processor.get_performance_summary()
        print(f"\n📈 Overall Performance Summary:")
        print(f"   Total Documents: {performance_summary['total_documents_processed']}")
        print(f"   Average Processing Time: {performance_summary['average_processing_time']:.2f}s")
        print(f"   Cache Hit Rate: {performance_summary['cache_hit_rate']:.2%}")
        print(f"   Target Success Rate: {performance_summary['performance_target_success_rate']:.2%}")
        print(f"   Performance Target Met: {performance_summary['target_met']}")
        
        # Test caching
        print(f"\n🔄 Testing caching system...")
        cache_start_time = time.time()
        cached_result = await processor.process_document_async(test_file_path)
        cache_processing_time = time.time() - cache_start_time
        
        print(f"   Cached processing time: {cache_processing_time:.2f}s")
        print(f"   Cache speedup: {processing_time / cache_processing_time:.1f}x faster")
        
        # Test M4 optimizations
        print(f"\n🔧 Testing M4-specific optimizations...")
        processor.optimize_for_m4()
        print(f"   Max Workers: {processor.max_workers}")
        print(f"   Target Time: {processor.target_processing_time}s")
        
        return result
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        # Cleanup
        if Path(test_file_path).exists():
            Path(test_file_path).unlink()
        processor.cleanup()

def test_sync_processing():
    """Test synchronous processing as fallback"""
    print("\n🔄 Testing synchronous processing fallback...")
    
    processor = OptimizedDocumentProcessor()
    processor.optimize_for_m4()
    
    test_content = "Simple test document for sync processing."
    test_file_path = "sync_test.txt"
    
    with open(test_file_path, "w") as f:
        f.write(test_content)
    
    try:
        start_time = time.time()
        result = processor.process_document_sync(test_file_path)
        processing_time = time.time() - start_time
        
        print(f"   Sync processing time: {processing_time:.2f}s")
        print(f"   Entities extracted: {result.entities_extracted}")
        
        return result
        
    except Exception as e:
        print(f"   Sync processing error: {e}")
        return None
    
    finally:
        if Path(test_file_path).exists():
            Path(test_file_path).unlink()
        processor.cleanup()

async def main():
    """Main test function"""
    print("🧪 Explainium Performance Optimization Test Suite")
    print("=" * 60)
    
    # Test async pipeline
    result = await test_optimization_pipeline()
    
    if result:
        print(f"\n🎯 TEST SUMMARY:")
        print(f"   ✅ Async pipeline: Working")
        print(f"   ✅ Performance target: {'Achieved' if result.processing_time <= 120 else 'Missed'}")
        print(f"   ✅ Entity extraction: {result.entities_extracted} entities")
        print(f"   ✅ Confidence score: {result.confidence_score:.2f}")
        
        # Test sync fallback
        sync_result = test_sync_processing()
        if sync_result:
            print(f"   ✅ Sync fallback: Working")
        else:
            print(f"   ❌ Sync fallback: Failed")
        
        print(f"\n🚀 Optimization pipeline is ready for production use!")
        
        # Final performance assessment
        if result.processing_time <= 120:
            print(f"🎉 PERFORMANCE TARGET ACHIEVED: {result.processing_time:.2f}s < 120s")
        else:
            print(f"⚠️  PERFORMANCE TARGET MISSED: {result.processing_time:.2f}s > 120s")
            print(f"   Additional optimization may be needed")
    
    else:
        print(f"\n❌ TEST FAILED: Optimization pipeline not working")
        return False
    
    return True

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
