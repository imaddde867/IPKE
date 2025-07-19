#!/usr/bin/env python3
"""
Test script for database enhancement structure and imports.
Tests that all modules can be imported and classes are properly defined.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all enhanced database modules can be imported."""
    print("=== Testing Database Module Imports ===")
    
    try:
        # Test database module imports
        from src.database.database import (
            get_db, get_db_session, init_db, get_connection_pool_status
        )
        print("✓ Database module imports successful")
        
        # Test migrations module imports
        from src.database.migrations import (
            Migration, MigrationManager, run_migrations, get_migration_status
        )
        print("✓ Migrations module imports successful")
        
        # Test enhanced CRUD imports
        from src.database.crud import (
            TransactionManager, BulkOperations, PaginatedQuery,
            create_document_with_entities, get_documents, 
            get_processing_statistics, update_document_status
        )
        print("✓ Enhanced CRUD module imports successful")
        
        # Test new model imports
        from src.database.models import ProcessingLog, SystemHealth
        print("✓ New model classes imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_class_structure():
    """Test that classes have expected methods and attributes."""
    print("\n=== Testing Class Structure ===")
    
    try:
        from src.database.migrations import Migration, MigrationManager
        from src.database.crud import TransactionManager, BulkOperations, PaginatedQuery
        
        # Test Migration class
        migration = Migration("001", "Test migration", "CREATE TABLE test();")
        assert hasattr(migration, 'version')
        assert hasattr(migration, 'description')
        assert hasattr(migration, 'up_sql')
        assert hasattr(migration, 'to_dict')
        assert hasattr(migration, 'from_dict')
        print("✓ Migration class structure correct")
        
        # Test MigrationManager class
        assert hasattr(MigrationManager, 'get_applied_migrations')
        assert hasattr(MigrationManager, 'apply_migration')
        assert hasattr(MigrationManager, 'rollback_migration')
        print("✓ MigrationManager class structure correct")
        
        # Test BulkOperations class
        assert hasattr(BulkOperations, 'bulk_insert_entities')
        assert hasattr(BulkOperations, 'bulk_insert_keyphrases')
        assert hasattr(BulkOperations, 'bulk_insert_equipment')
        print("✓ BulkOperations class structure correct")
        
        # Test PaginatedQuery class
        assert hasattr(PaginatedQuery, 'paginate')
        print("✓ PaginatedQuery class structure correct")
        
        return True
        
    except Exception as e:
        print(f"✗ Class structure test failed: {e}")
        return False

def test_configuration():
    """Test that configuration includes new database settings."""
    print("\n=== Testing Configuration ===")
    
    try:
        from src.config import config_manager
        
        db_config = config_manager.config.database
        
        # Check new connection pooling settings
        assert hasattr(db_config, 'pool_size')
        assert hasattr(db_config, 'max_overflow')
        assert hasattr(db_config, 'pool_recycle')
        assert hasattr(db_config, 'pool_pre_ping')
        assert hasattr(db_config, 'connect_timeout')
        
        print(f"✓ Database pool size: {db_config.pool_size}")
        print(f"✓ Database max overflow: {db_config.max_overflow}")
        print(f"✓ Database pool recycle: {db_config.pool_recycle}")
        print(f"✓ Database pool pre-ping: {db_config.pool_pre_ping}")
        print(f"✓ Database connect timeout: {db_config.connect_timeout}")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_migration_definitions():
    """Test that initial migrations are properly defined."""
    print("\n=== Testing Migration Definitions ===")
    
    try:
        from src.database.migrations import INITIAL_MIGRATIONS
        
        assert len(INITIAL_MIGRATIONS) > 0, "No initial migrations defined"
        
        for migration in INITIAL_MIGRATIONS:
            assert migration.version, f"Migration missing version: {migration}"
            assert migration.description, f"Migration missing description: {migration}"
            assert migration.up_sql, f"Migration missing up_sql: {migration}"
            print(f"✓ Migration {migration.version}: {migration.description}")
        
        print(f"✓ Total migrations defined: {len(INITIAL_MIGRATIONS)}")
        return True
        
    except Exception as e:
        print(f"✗ Migration definitions test failed: {e}")
        return False

def test_model_enhancements():
    """Test that models have been enhanced with new fields."""
    print("\n=== Testing Model Enhancements ===")
    
    try:
        from src.database.models import Document, ProcessingLog, SystemHealth
        
        # Check Document model enhancements
        doc_columns = [attr for attr in dir(Document) if not attr.startswith('_')]
        expected_fields = [
            'processing_duration_seconds', 'retry_count', 'last_error',
            'processing_started_at', 'processing_completed_at'
        ]
        
        for field in expected_fields:
            assert field in doc_columns, f"Document model missing field: {field}"
        
        print("✓ Document model enhanced with processing fields")
        
        # Check new model classes
        processing_log_columns = [attr for attr in dir(ProcessingLog) if not attr.startswith('_')]
        expected_log_fields = ['document_id', 'step_name', 'status', 'start_time', 'end_time']
        
        for field in expected_log_fields:
            assert field in processing_log_columns, f"ProcessingLog model missing field: {field}"
        
        print("✓ ProcessingLog model properly defined")
        
        # Check SystemHealth model
        health_columns = [attr for attr in dir(SystemHealth) if not attr.startswith('_')]
        expected_health_fields = ['service_name', 'status', 'response_time_ms', 'checked_at']
        
        for field in expected_health_fields:
            assert field in health_columns, f"SystemHealth model missing field: {field}"
        
        print("✓ SystemHealth model properly defined")
        
        return True
        
    except Exception as e:
        print(f"✗ Model enhancements test failed: {e}")
        return False

def main():
    """Run all structure tests."""
    print("Database Enhancement Structure Tests")
    print("=" * 50)
    
    tests = [
        ("Module Imports", test_imports),
        ("Class Structure", test_class_structure),
        ("Configuration", test_configuration),
        ("Migration Definitions", test_migration_definitions),
        ("Model Enhancements", test_model_enhancements)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = {"success": success}
        except Exception as e:
            results[test_name] = {"success": False, "error": str(e)}
    
    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    for test_name, result in results.items():
        status = "PASS" if result["success"] else "FAIL"
        print(f"{test_name:<25} {status}")
        if not result["success"] and "error" in result:
            print(f"  Error: {result['error']}")
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r["success"])
    
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("All database enhancement structure tests passed!")
        return 0
    else:
        print("Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    exit(main())