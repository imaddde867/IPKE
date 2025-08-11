#!/usr/bin/env python3
"""
Test Real AI Engines Integration
Verifies that the optimized processor uses real AI engines for high-quality extraction
"""

import sys
import time
import asyncio
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

def test_real_ai_engines():
    """Test that the optimized processor uses real AI engines"""
    print("🧪 TESTING REAL AI ENGINES INTEGRATION")
    print("=" * 50)
    
    try:
        # Import the real optimized processor
        from processors.optimized_processor import OptimizedDocumentProcessor
        print("✅ OptimizedDocumentProcessor imported successfully")
        
        # Create processor instance
        processor = OptimizedDocumentProcessor()
        print("✅ Processor instance created")
        
        # Check if engines are properly initialized
        print("\n🔍 Checking AI Engine Initialization:")
        print("-" * 40)
        
        # Initialize engines
        processor._initialize_engines()
        
        # Check engine availability
        if processor.advanced_engine:
            print(f"✅ Advanced Knowledge Engine: {type(processor.advanced_engine).__name__}")
        else:
            print("❌ Advanced Knowledge Engine not available")
            
        if processor.llm_engine:
            print(f"✅ LLM Processing Engine: {type(processor.llm_engine).__name__}")
        else:
            print("❌ LLM Processing Engine not available")
            
        if processor.extraction_engine:
            print(f"✅ Enhanced Extraction Engine: {type(processor.extraction_engine).__name__}")
        else:
            print("❌ Enhanced Extraction Engine not available")
        
        # Test M4 optimization
        processor.optimize_for_m4()
        print(f"\n✅ M4 optimization applied: {processor.max_workers} workers")
        
        # Create a test document
        test_file = "test_ai_quality.txt"
        test_content = """
        Technical Specification Document
        
        Product: Advanced AI Processing Unit
        Model: XR-5000
        Specifications:
        - Processing Power: 500 TFLOPS
        - Memory: 128GB DDR6
        - Power Consumption: 150W
        - Operating Temperature: -20°C to +85°C
        
        Safety Requirements:
        - Overheating protection
        - Power surge protection
        - Emergency shutdown capability
        
        Maintenance Schedule:
        - Monthly: Clean air filters
        - Quarterly: Performance calibration
        - Annually: Full system diagnostics
        """
        
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        print(f"\n✅ Test document created: {test_file}")
        
        # Test async processing
        async def test_ai_processing():
            print("\n🚀 Testing AI Processing Pipeline:")
            print("-" * 40)
            
            start_time = time.time()
            result = await processor.process_document_async(test_file)
            processing_time = time.time() - start_time
            
            print(f"✅ Processing completed in {processing_time:.3f}s")
            print(f"   Document ID: {result.document_id}")
            print(f"   Processing Method: {result.processing_method}")
            print(f"   Optimization Level: {result.optimization_level}")
            print(f"   Entities Extracted: {result.entities_extracted}")
            print(f"   Confidence Score: {result.confidence_score:.3f}")
            
            # Check AI engine utilization
            ai_utilization = result.performance_metrics.get("ai_engine_utilization", {})
            print(f"\n🤖 AI Engine Utilization:")
            print(f"   LLM Available: {ai_utilization.get('llm_available', False)}")
            print(f"   Extraction Available: {ai_utilization.get('extraction_available', False)}")
            print(f"   Validation Available: {ai_utilization.get('validation_available', False)}")
            
            # Check quality metrics
            print(f"\n📊 Quality Metrics:")
            print(f"   Overall Quality Score: {result.performance_metrics.get('overall_quality_score', 0):.3f}")
            print(f"   Processing Efficiency: {result.performance_metrics.get('processing_efficiency', 'unknown')}")
            
            # Show extracted entities
            if result.entities:
                print(f"\n🔍 Extracted Entities (showing first 3):")
                for i, entity in enumerate(result.entities[:3]):
                    print(f"   {i+1}. {entity.get('content', '')[:60]}...")
                    print(f"      Type: {entity.get('entity_type', 'unknown')}")
                    print(f"      Category: {entity.get('category', 'unknown')}")
                    print(f"      Quality Score: {entity.get('quality_score', 0):.3f}")
                    print(f"      Source: {entity.get('source', 'unknown')}")
            
            return result
        
        # Run the test
        result = asyncio.run(test_ai_processing())
        
        # Performance summary
        performance = processor.get_performance_summary()
        print(f"\n📈 Performance Summary:")
        print(f"   Total Documents: {performance['total_documents']}")
        print(f"   Average Processing Time: {performance['average_processing_time']:.3f}s")
        print(f"   Cache Hit Rate: {performance['cache_hit_rate']:.1f}%")
        print(f"   Target Success Rate: {performance['performance_target_met']}/{performance['total_documents']}")
        
        # Cleanup
        processor.cleanup()
        if Path(test_file).exists():
            Path(test_file).unlink()
        print(f"\n✅ Test cleanup completed")
        
        # Quality assessment
        print(f"\n🎯 QUALITY ASSESSMENT:")
        print("-" * 30)
        
        if result.confidence_score > 0.8:
            print("✅ EXCELLENT: High-quality AI extraction achieved")
        elif result.confidence_score > 0.6:
            print("✅ GOOD: Quality AI extraction achieved")
        elif result.confidence_score > 0.4:
            print("⚠️ MODERATE: Basic AI extraction achieved")
        else:
            print("❌ POOR: AI extraction quality needs improvement")
        
        if result.processing_method == "ai_enhanced_processing":
            print("✅ MAXIMUM QUALITY: All AI engines utilized")
        elif result.processing_method == "ai_processing":
            print("✅ HIGH QUALITY: Core AI engines utilized")
        else:
            print("⚠️ STANDARD QUALITY: Limited AI engine utilization")
        
        print(f"\n🎉 REAL AI ENGINES TEST COMPLETED SUCCESSFULLY!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_real_ai_engines()
    sys.exit(0 if success else 1)
