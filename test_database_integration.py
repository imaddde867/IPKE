#!/usr/bin/env python3
"""
Integration test for database enhancements.
This should be run within the Docker environment where the database is available.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.database.database import get_db_session, get_connection_pool_status
from src.database.migrations import run_migrations, get_migration_status
from src.database.crud import (
    create_document_with_entities, 
    BulkOperations, 
    get_processing_statistics,
    update_document_status,
    get_documents
)
from src.logging_config import get_logger
import time

logger = get_logger(__name__)

def test_basic_connectivity():
    """Test basic database connectivity."""
    print("=== Testing Database Connectivity ===")
    
    try:
        with get_db_session() as db:
            result = db.execute("SELECT 1 as test").fetchone()
            assert result[0] == 1
            print("✓ Database connection successful")
            return True
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False

def test_migrations():
    """Test migration system."""
    print("\n=== Testing Migrations ===")
    
    try:
        # Get initial status
        initial_status = get_migration_status()
        print(f"✓ Initial migration status retrieved: {len(initial_status.get('applied_migrations', []))} applied")
        
        # Run migrations
        applied_count = run_migrations()
        print(f"✓ Migrations completed: {applied_count} new migrations applied")
        
        # Get final status
        final_status = get_migration_status()
        print(f"✓ Final migration status: {len(final_status.get('applied_migrations', []))} total applied")
        
        return True
    except Exception as e:
        print(f"✗ Migration test failed: {e}")
        return False

def test_connection_pool():
    """Test connection pool functionality."""
    print("\n=== Testing Connection Pool ===")
    
    try:
        pool_status = get_connection_pool_status()
        print(f"✓ Pool status retrieved: {pool_status}")
        
        # Test multiple sessions
        sessions_created = 0
        for i in range(3):
            with get_db_session() as db:
                db.execute("SELECT 1")
                sessions_created += 1
        
        print(f"✓ Created {sessions_created} database sessions successfully")
        return True
    except Exception as e:
        print(f"✗ Connection pool test failed: {e}")
        return False

def test_enhanced_crud():
    """Test enhanced CRUD operations."""
    print("\n=== Testing Enhanced CRUD Operations ===")
    
    try:
        with get_db_session() as db:
            # Test document creation with entities
            doc = create_document_with_entities(
                db=db,
                filename="test_integration.pdf",
                file_type="pdf",
                extracted_text="Integration test document",
                metadata_json={"test": "integration"},
                classification_category="test",
                classification_score=0.95,
                status="processing",
                document_sections={"section1": "test content"},
                entities=[
                    {"text": "Test Entity", "entity_type": "TEST", "score": 0.9, "start_char": 0, "end_char": 11}
                ],
                keyphrases=["integration", "test", "database"]
            )
            
            print(f"✓ Document created with ID: {doc.id}")
            
            # Test status update
            success = update_document_status(db, doc.id, "completed")
            print(f"✓ Status update successful: {success}")
            
            # Test pagination
            docs_page = get_documents(db, page=1, per_page=5)
            print(f"✓ Paginated query successful: {len(docs_page['items'])} items, {docs_page['total']} total")
            
            # Test statistics
            stats = get_processing_statistics(db)
            print(f"✓ Statistics retrieved: {stats.get('total_entities_extracted', 0)} total entities")
            
        return True
    except Exception as e:
        print(f"✗ Enhanced CRUD test failed: {e}")
        return False

def test_bulk_operations():
    """Test bulk operations."""
    print("\n=== Testing Bulk Operations ===")
    
    try:
        with get_db_session() as db:
            # Create a test document
            doc = create_document_with_entities(
                db=db,
                filename="test_bulk_integration.pdf",
                file_type="pdf",
                extracted_text="Bulk operations test document",
                metadata_json={"bulk_test": True},
                classification_category="test",
                classification_score=0.85,
                status="processing",
                document_sections={"section1": "bulk content"}
            )
            
            # Test bulk entity insertion
            entities = [
                {"text": f"Bulk Entity {i}", "entity_type": "BULK_TEST", "score": 0.8, "start_char": i*15, "end_char": i*15+12}
                for i in range(5)
            ]
            
            bulk_entities = BulkOperations.bulk_insert_entities(db, doc.id, entities)
            print(f"✓ Bulk inserted {len(bulk_entities)} entities")
            
            # Test bulk keyphrase insertion
            keyphrases = [f"bulk_phrase_{i}" for i in range(3)]
            bulk_phrases = BulkOperations.bulk_insert_keyphrases(db, doc.id, keyphrases)
            print(f"✓ Bulk inserted {len(bulk_phrases)} keyphrases")
            
        return True
    except Exception as e:
        print(f"✗ Bulk operations test failed: {e}")
        return False

def main():
    """Run integration tests."""
    print("Database Enhancement Integration Tests")
    print("=" * 50)
    print("Note: This test requires a running PostgreSQL database")
    print("=" * 50)
    
    tests = [
        ("Database Connectivity", test_basic_connectivity),
        ("Migrations", test_migrations),
        ("Connection Pool", test_connection_pool),
        ("Enhanced CRUD", test_enhanced_crud),
        ("Bulk Operations", test_bulk_operations)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            start_time = time.time()
            success = test_func()
            duration = time.time() - start_time
            results[test_name] = {"success": success, "duration": duration}
        except Exception as e:
            results[test_name] = {"success": False, "error": str(e), "duration": 0}
    
    # Print summary
    print("\n" + "=" * 50)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 50)
    
    for test_name, result in results.items():
        status = "PASS" if result["success"] else "FAIL"
        duration = result.get("duration", 0)
        print(f"{test_name:<25} {status:<6} ({duration:.3f}s)")
        if not result["success"] and "error" in result:
            print(f"  Error: {result['error']}")
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r["success"])
    
    print(f"\nTotal: {passed_tests}/{total_tests} integration tests passed")
    
    if passed_tests == total_tests:
        print("All database enhancement integration tests passed!")
        return 0
    else:
        print("Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    exit(main())