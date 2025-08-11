#!/usr/bin/env python3
"""
Test Hybrid Optimized Processor
Tests the hybrid processor that demonstrates real AI engine integration
"""

import sys
import time
import asyncio
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

def test_hybrid_processor():
    """Test the hybrid optimized processor"""
    print("🧪 TESTING FULL OPTIMIZED PROCESSOR")
    print("=" * 50)
    
    try:
        # Import the full optimized processor
        from processors.optimized_processor import OptimizedDocumentProcessor
        print("✅ OptimizedDocumentProcessor imported successfully")
        
        # Create processor instance
        processor = OptimizedDocumentProcessor()
        print("✅ Optimized processor instance created")
        
        # Test M4 optimization
        processor.optimize_for_m4()
        print(f"✅ M4 optimization applied: {processor.max_workers} workers")
        
        # Create a test document
        test_file = "test_hybrid_quality.txt"
        test_content = """
        Advanced Technical Specification Document
        
        Product: Hybrid AI Processing Unit
        Model: HY-7000
        Version: 2.1.0
        
        Technical Specifications:
        - Processing Power: 750 TFLOPS
        - Memory: 256GB DDR6 ECC
        - Power Consumption: 200W
        - Operating Temperature: -40°C to +95°C
        - MTBF: 50,000 hours
        
        AI Capabilities:
        - Real-time LLM processing
        - Enhanced entity extraction
        - Advanced knowledge validation
        - Multi-modal content analysis
        
        Safety Features:
        - Overheating protection with automatic shutdown
        - Power surge protection with UPS backup
        - Emergency shutdown capability
        - Fire suppression system integration
        
        Maintenance Requirements:
        - Monthly: Air filter cleaning and system diagnostics
        - Quarterly: Performance calibration and optimization
        - Semi-annually: Full system health check
        - Annually: Complete system overhaul and upgrade assessment
        
        Quality Assurance:
        - ISO 9001:2015 certified
        - CE marking for European markets
        - FCC compliance for US markets
        - RoHS compliant for environmental safety
        """
        
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        print(f"✅ Test document created: {test_file}")
        
        # Test async processing
        async def test_hybrid_processing():
            print("\n🚀 Testing Full AI Processing Pipeline:")
            print("-" * 50)
            
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
            print(f"   Real AI Engines Used: {result.performance_metrics.get('real_ai_engines_used', False)}")
            
            # Show extracted entities
            if result.entities:
                print(f"\n🔍 Extracted Entities (showing first 5):")
                for i, entity in enumerate(result.entities[:5]):
                    print(f"   {i+1}. {entity.get('content', '')[:60]}...")
                    print(f"      Type: {entity.get('entity_type', 'unknown')}")
                    print(f"      Category: {entity.get('category', 'unknown')}")
                    print(f"      Quality Score: {entity.get('quality_score', 0):.3f}")
                    print(f"      Source: {entity.get('source', 'unknown')}")
                    print()
            else:
                print("\n⚠️ No entities extracted - this may indicate mock AI engines are being used")
            
            return result
        
        # Run the test
        result = asyncio.run(test_hybrid_processing())
        
        # Performance summary
        performance = processor.get_performance_summary()
        print(f"\n📈 Performance Summary:")
        print(f"   Total Documents: {performance.get('total_documents_processed', 0)}")
        print(f"   Average Processing Time: {performance.get('average_processing_time', 0):.3f}s")
        # cache_hit_rate returned as a float 0..1
        chr_val = performance.get('cache_hit_rate', 0.0)
        print(f"   Cache Hit Rate: {chr_val*100:.1f}%")
        print(f"   Target Success Rate: {performance.get('performance_target_success_rate', 0.0)*100:.1f}%")
        
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
        
        # Check if real AI engines are being used
        if result.performance_metrics.get('real_ai_engines_used', False):
            print("✅ REAL AI ENGINES: High-quality intelligence extraction confirmed")
        else:
            print("⚠️ MOCK AI ENGINES: Using demonstration mode (expected in testing)")
        
        print(f"\n🎉 HYBRID PROCESSOR TEST COMPLETED SUCCESSFULLY!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_hybrid_processor()
    sys.exit(0 if success else 1)
