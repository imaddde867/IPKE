#!/usr/bin/env python3
"""
Minimal Performance Optimization Test
Tests the optimization structure without external dependencies
"""

import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test if we can import the simplified optimized components"""
    print("🧪 Testing Import Structure")
    print("=" * 40)
    
    try:
        # Test simplified optimized processor import
        from processors.optimized_processor_simple import OptimizedDocumentProcessorSimple
        print("✅ OptimizedDocumentProcessorSimple imported successfully")
        
        # Test simplified engines
        from processors.optimized_processor_simple import SimpleLLMEngine, SimpleExtractionEngine
        print("✅ SimpleLLMEngine imported successfully")
        print("✅ SimpleExtractionEngine imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_processor_structure():
    """Test the simplified processor structure and methods"""
    print("\n🔍 Testing Simplified Processor Structure")
    print("=" * 50)
    
    try:
        from processors.optimized_processor_simple import OptimizedDocumentProcessorSimple
        
        # Create processor instance
        processor = OptimizedDocumentProcessorSimple()
        print("✅ Simplified processor instance created")
        
        # Check methods exist
        methods = [
            'process_document_async',
            'process_document_sync', 
            'optimize_for_m4',
            'get_performance_summary',
            'cleanup'
        ]
        
        for method in methods:
            if hasattr(processor, method):
                print(f"✅ Method {method} exists")
            else:
                print(f"❌ Method {method} missing")
        
        # Test M4 optimization
        processor.optimize_for_m4()
        print(f"✅ M4 optimization applied: {processor.max_workers} workers")
        
        # Test performance summary
        summary = processor.get_performance_summary()
        print(f"✅ Performance summary: {summary['status']}")
        
        # Cleanup
        processor.cleanup()
        print("✅ Cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Structure test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_engine_structure():
    """Test the simplified engine structures"""
    print("\n🔍 Testing Simplified Engine Structures")
    print("=" * 50)
    
    try:
        # Test simplified engines
        from processors.optimized_processor_simple import SimpleLLMEngine, SimpleExtractionEngine
        
        # Test LLM engine
        llm_engine = SimpleLLMEngine()
        print("✅ Simple LLM engine created")
        
        # Check LLM methods
        llm_methods = ['process_document_optimized', 'cleanup']
        for method in llm_methods:
            if hasattr(llm_engine, method):
                print(f"✅ LLM method {method} exists")
            else:
                print(f"❌ LLM method {method} missing")
        
        llm_engine.cleanup()
        print("✅ Simple LLM engine cleanup completed")
        
        # Test extraction engine
        extraction_engine = SimpleExtractionEngine()
        print("✅ Simple extraction engine created")
        
        # Check extraction methods
        extraction_methods = ['extract_comprehensive_knowledge', 'cleanup']
        for method in extraction_methods:
            if hasattr(extraction_engine, method):
                print(f"✅ Extraction method {method} exists")
            else:
                print(f"❌ Extraction method {method} missing")
        
        extraction_engine.cleanup()
        print("✅ Simple extraction engine cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Engine test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_async_processing():
    """Test the async processing pipeline"""
    print("\n🔍 Testing Async Processing Pipeline")
    print("=" * 50)
    
    try:
        from processors.optimized_processor_simple import OptimizedDocumentProcessorSimple
        import asyncio
        
        # Create processor
        processor = OptimizedDocumentProcessorSimple()
        processor.optimize_for_m4()
        
        # Create a temporary test file
        test_file = "test_document.txt"
        with open(test_file, 'w') as f:
            f.write("This is a test document for performance optimization testing.")
        
        print("✅ Test file created")
        
        # Test async processing
        async def test_processing():
            result = await processor.process_document_async(test_file)
            return result
        
        # Run async test
        result = asyncio.run(test_processing())
        print(f"✅ Async processing completed: {result.document_id}")
        print(f"   Processing time: {result.processing_time:.3f}s")
        print(f"   Entities extracted: {result.entities_extracted}")
        print(f"   Confidence: {result.confidence_score:.2f}")
        
        # Test sync wrapper
        result_sync = processor.process_document_sync(test_file)
        print(f"✅ Sync processing completed: {result_sync.document_id}")
        
        # Check performance
        performance = processor.get_performance_summary()
        print(f"✅ Performance tracking: {performance['total_documents_processed']} documents")
        
        # Cleanup
        processor.cleanup()
        import os
        if os.path.exists(test_file):
            os.remove(test_file)
        print("✅ Test cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Async processing test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🧪 MINIMAL Explainium Performance Optimization Test")
    print("=" * 60)
    
    # Test imports
    imports_ok = test_imports()
    
    if not imports_ok:
        print("\n❌ Import tests failed. Check dependencies.")
        return False
    
    # Test processor structure
    processor_ok = test_processor_structure()
    
    # Test engine structures
    engines_ok = test_engine_structure()
    
    # Test async processing
    async_ok = test_async_processing()
    
    # Summary
    print("\n📊 TEST SUMMARY")
    print("=" * 30)
    
    if imports_ok and processor_ok and engines_ok and async_ok:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Import structure: Working")
        print("✅ Processor structure: Working") 
        print("✅ Engine structures: Working")
        print("✅ Async processing: Working")
        print("\n🚀 Optimization pipeline is structurally ready!")
        print("📝 Note: This is the simplified version for testing")
        print("🔧 Full version available in optimized_processor.py")
        return True
    else:
        print("❌ SOME TESTS FAILED")
        print(f"   Imports: {'✅' if imports_ok else '❌'}")
        print(f"   Processor: {'✅' if processor_ok else '❌'}")
        print(f"   Engines: {'✅' if engines_ok else '❌'}")
        print(f"   Async Processing: {'✅' if async_ok else '❌'}")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
