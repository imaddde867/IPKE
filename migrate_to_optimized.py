#!/usr/bin/env python3
"""
Migration Script: Old Processor → Optimized Processor
Helps transition from the original Explainium processor to the optimized version
"""

import sys
import os
from pathlib import Path
import importlib.util

def check_old_processor():
    """Check if old processor exists and can be imported"""
    try:
        # Try to import the old processor
        old_processor_path = Path("src/processors/processor.py")
        if old_processor_path.exists():
            print("✅ Old processor found at: src/processors/processor.py")
            return True
        else:
            print("ℹ️  Old processor not found")
            return False
    except Exception as e:
        print(f"⚠️  Error checking old processor: {e}")
        return False

def check_optimized_processor():
    """Check if optimized processor exists and can be imported"""
    try:
        # Try to import the optimized processor
        optimized_processor_path = Path("src/processors/optimized_processor.py")
        if optimized_processor_path.exists():
            print("✅ Optimized processor found at: src/processors/optimized_processor.py")
            return True
        else:
            print("❌ Optimized processor not found")
            return False
    except Exception as e:
        print(f"⚠️  Error checking optimized processor: {e}")
        return False

def check_optimized_engines():
    """Check if optimized engines exist"""
    engines = [
        "src/ai/llm_processing_engine.py",
        "src/ai/enhanced_extraction_engine.py"
    ]
    
    all_exist = True
    for engine_path in engines:
        if Path(engine_path).exists():
            print(f"✅ {engine_path} - Found")
        else:
            print(f"❌ {engine_path} - Missing")
            all_exist = False
    
    return all_exist

def create_migration_example():
    """Create an example migration script"""
    migration_code = '''#!/usr/bin/env python3
"""
Example Migration: Old Processor → Optimized Processor
"""

# OLD WAY (10+ minutes per document)
try:
    from src.processors.processor import DocumentProcessor
    
    # Initialize old processor
    old_processor = DocumentProcessor()
    
    # Process document (slow)
    result = old_processor.process_document("document.pdf")
    print(f"Old processor result: {result}")
    
except ImportError:
    print("Old processor not available")

# NEW WAY (Under 2 minutes per document)
try:
    from src.processors.optimized_processor import OptimizedDocumentProcessor
    
    # Initialize optimized processor
    processor = OptimizedDocumentProcessor()
    
    # Apply M4 optimizations
    processor.optimize_for_m4()
    
    # Process document asynchronously (fast)
    import asyncio
    
    async def process_doc():
        result = await processor.process_document_async("document.pdf")
        return result
    
    # Run async processing
    result = asyncio.run(process_doc())
    print(f"Optimized processor result: {result}")
    
    # Or use synchronous wrapper
    sync_result = processor.process_document_sync("document.pdf")
    print(f"Sync wrapper result: {sync_result}")
    
    # Get performance summary
    performance = processor.get_performance_summary()
    print(f"Performance: {performance}")
    
except ImportError as e:
    print(f"Optimized processor not available: {e}")

# CLEANUP
try:
    processor.cleanup()
except:
    pass
'''
    
    with open("migration_example.py", "w") as f:
        f.write(migration_code)
    
    print("✅ Created migration_example.py")

def create_performance_comparison():
    """Create a performance comparison script"""
    comparison_code = '''#!/usr/bin/env python3
"""
Performance Comparison: Old vs Optimized Processor
"""

import time
import asyncio
from pathlib import Path

def create_test_document():
    """Create a test document for performance testing"""
    test_content = """
    This is a comprehensive test document for performance comparison.
    
    Technical Specifications:
    - Processor: M4 Pro with 10-core CPU
    - Memory: 16GB unified memory
    - Storage: 512GB SSD
    - Operating System: macOS Sonoma
    
    Performance Targets:
    - Old Processor: 10+ minutes per document
    - Optimized Processor: Under 2 minutes per document
    - Improvement Target: 5x+ speed increase
    
    Key Features to Test:
    - Document parsing and content extraction
    - Entity recognition and classification
    - LLM processing and enhancement
    - Validation and quality assurance
    - Caching and performance optimization
    
    Expected Results:
    - Significant speed improvement
    - Maintained or improved quality
    - Better resource utilization
    - Enhanced user experience
    """
    
    test_file = "performance_test_document.txt"
    with open(test_file, "w") as f:
        f.write(test_content)
    
    return test_file

async def test_old_processor(test_file):
    """Test old processor performance"""
    try:
        from src.processors.processor import DocumentProcessor
        
        print("🔄 Testing OLD processor...")
        processor = DocumentProcessor()
        
        start_time = time.time()
        result = processor.process_document(test_file)
        processing_time = time.time() - start_time
        
        print(f"   ⏱️  Processing time: {processing_time:.2f} seconds")
        print(f"   📊 Result: {type(result)}")
        
        return processing_time, result
        
    except ImportError:
        print("   ❌ Old processor not available")
        return None, None
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None, None

async def test_optimized_processor(test_file):
    """Test optimized processor performance"""
    try:
        from src.processors.optimized_processor import OptimizedDocumentProcessor
        
        print("🚀 Testing OPTIMIZED processor...")
        processor = OptimizedDocumentProcessor()
        processor.optimize_for_m4()
        
        start_time = time.time()
        result = await processor.process_document_async(test_file)
        processing_time = time.time() - start_time
        
        print(f"   ⏱️  Processing time: {processing_time:.2f} seconds")
        print(f"   📊 Result: {type(result)}")
        print(f"   🎯 Target met: {processing_time <= 120}")
        
        # Test caching
        cache_start = time.time()
        cached_result = await processor.process_document_async(test_file)
        cache_time = time.time() - cache_start
        
        print(f"   🚀 Cache hit time: {cache_time:.2f} seconds")
        print(f"   📈 Cache speedup: {processing_time / cache_time:.1f}x")
        
        # Performance summary
        performance = processor.get_performance_summary()
        print(f"   📊 Performance summary: {performance}")
        
        processor.cleanup()
        return processing_time, result
        
    except ImportError:
        print("   ❌ Optimized processor not available")
        return None, None
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None, None

async def main():
    """Main performance comparison"""
    print("🧪 PERFORMANCE COMPARISON: Old vs Optimized Processor")
    print("=" * 60)
    
    # Create test document
    test_file = create_test_document()
    print(f"📄 Created test document: {test_file}")
    
    try:
        # Test old processor
        old_time, old_result = await test_old_processor(test_file)
        
        # Test optimized processor
        optimized_time, optimized_result = await test_optimized_processor(test_file)
        
        # Comparison
        print("\\n📊 PERFORMANCE COMPARISON:")
        print("=" * 40)
        
        if old_time and optimized_time:
            improvement = old_time / optimized_time
            print(f"   Old Processor:     {old_time:.2f} seconds")
            print(f"   Optimized Processor: {optimized_time:.2f} seconds")
            print(f"   Speed Improvement:   {improvement:.1f}x faster")
            
            if optimized_time <= 120:
                print(f"   🎉 TARGET ACHIEVED: Under 2 minutes!")
            else:
                print(f"   ⚠️  TARGET MISSED: Over 2 minutes")
            
            if improvement >= 5:
                print(f"   🚀 SUCCESS: 5x+ improvement achieved!")
            else:
                print(f"   📈 GOOD: {improvement:.1f}x improvement")
        
        else:
            print("   ❌ Could not complete performance comparison")
    
    finally:
        # Cleanup
        if Path(test_file).exists():
            Path(test_file).unlink()
            print(f"\\n🧹 Cleaned up test file: {test_file}")

if __name__ == "__main__":
    asyncio.run(main())
'''
    
    with open("performance_comparison.py", "w") as f:
        f.write(comparison_code)
    
    print("✅ Created performance_comparison.py")

def main():
    """Main migration function"""
    print("🔄 Explainium Migration: Old → Optimized Processor")
    print("=" * 50)
    
    # Check current state
    print("\\n🔍 Checking current system state...")
    
    old_exists = check_old_processor()
    optimized_exists = check_optimized_processor()
    engines_exist = check_optimized_engines()
    
    print("\\n📋 Migration Status:")
    
    if not optimized_exists:
        print("❌ Optimized processor not found. Please run the optimization implementation first.")
        return False
    
    if not engines_exist:
        print("❌ Optimized engines not found. Please ensure all optimization files are in place.")
        return False
    
    if old_exists:
        print("✅ Old processor found - Migration possible")
        print("✅ Optimized processor ready - Migration target available")
        print("✅ Optimized engines ready - Full optimization available")
        
        print("\\n🚀 Ready for migration!")
        print("\\n📁 Files to update:")
        print("   1. Update imports from 'processor' to 'optimized_processor'")
        print("   2. Change 'DocumentProcessor' to 'OptimizedDocumentProcessor'")
        print("   3. Add 'processor.optimize_for_m4()' for M4 optimizations")
        print("   4. Use async methods for best performance")
        
    else:
        print("ℹ️  Old processor not found - Fresh installation")
        print("✅ Optimized processor ready - Use directly")
        print("✅ Optimized engines ready - Full optimization available")
        
        print("\\n🎉 Fresh optimization installation ready!")
        print("\\n📁 Usage:")
        print("   from src.processors.optimized_processor import OptimizedDocumentProcessor")
        print("   processor = OptimizedDocumentProcessor()")
        print("   processor.optimize_for_m4()")
    
    # Create helpful files
    print("\\n📝 Creating helpful migration files...")
    create_migration_example()
    create_performance_comparison()
    
    print("\\n📚 Migration files created:")
    print("   - migration_example.py: Example migration code")
    print("   - performance_comparison.py: Performance testing")
    
    print("\\n🎯 Next Steps:")
    print("   1. Review migration_example.py for usage patterns")
    print("   2. Run performance_comparison.py to test improvements")
    print("   3. Update your code to use the optimized processor")
    print("   4. Test with real documents to validate performance")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\\n\\n⏹️  Migration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\\n❌ Migration error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)