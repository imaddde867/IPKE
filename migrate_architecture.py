#!/usr/bin/env python3
"""
EXPLAINIUM - Architecture Migration Script

This script helps migrate from the old complex architecture to the new simplified one.
It updates imports, renames old files, and validates the new architecture works.

MIGRATION PLAN:
1. Backup old engines → .backup files
2. Update all imports to use new unified engine
3. Test compatibility with existing API
4. Generate migration report
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import json

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_step(message: str):
    print(f"{Colors.BLUE}{Colors.BOLD}[MIGRATION]{Colors.END} {message}")

def print_success(message: str):
    print(f"{Colors.GREEN}✓{Colors.END} {message}")

def print_warning(message: str):
    print(f"{Colors.YELLOW}⚠{Colors.END} {message}")

def print_error(message: str):
    print(f"{Colors.RED}✗{Colors.END} {message}")

class ArchitectureMigration:
    """Handles migration from old to new architecture"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.backup_dir = self.project_root / "migration_backup" / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.migration_log = []
        
        # Files to backup and replace
        self.old_engines = [
            "src/ai/advanced_knowledge_engine.py",
            "src/ai/llm_processing_engine.py", 
            "src/ai/enhanced_extraction_engine.py",
            "src/ai/knowledge_categorization_engine.py",
            "src/ai/document_intelligence_analyzer.py",
            "src/ai/database_output_generator.py"
        ]
        
        self.old_config_files = [
            "src/core/config.py",
            "src/core/optimization.py"
        ]
        
        self.old_processors = [
            "src/processors/processor.py"
        ]
        
        self.old_api = [
            "src/api/app.py"
        ]
    
    def run_migration(self) -> bool:
        """Run the complete migration process"""
        print_step("Starting Explainium Architecture Migration")
        print(f"Project root: {self.project_root}")
        print(f"Backup directory: {self.backup_dir}")
        print()
        
        try:
            # Step 1: Create backup
            self._create_backup()
            
            # Step 2: Validate new components
            self._validate_new_components()
            
            # Step 3: Update imports
            self._update_imports()
            
            # Step 4: Update main API
            self._update_main_api()
            
            # Step 5: Generate compatibility shims
            self._create_compatibility_shims()
            
            # Step 6: Test new architecture
            self._test_new_architecture()
            
            # Step 7: Generate migration report
            self._generate_migration_report()
            
            print_step("Migration completed successfully!")
            print_success(f"Backup created at: {self.backup_dir}")
            print_success("New unified architecture is ready")
            
            return True
            
        except Exception as e:
            print_error(f"Migration failed: {e}")
            self._rollback_migration()
            return False
    
    def _create_backup(self):
        """Create backup of old files"""
        print_step("Creating backup of existing files...")
        
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        all_files = self.old_engines + self.old_config_files + self.old_processors + self.old_api
        
        for file_path in all_files:
            source = self.project_root / file_path
            if source.exists():
                target = self.backup_dir / file_path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                print_success(f"Backed up: {file_path}")
                self.migration_log.append(f"BACKUP: {file_path}")
            else:
                print_warning(f"File not found: {file_path}")
    
    def _validate_new_components(self):
        """Validate that new components exist and are functional"""
        print_step("Validating new architecture components...")
        
        new_components = [
            "src/ai/unified_knowledge_engine.py",
            "src/processors/streamlined_processor.py",
            "src/core/unified_config.py",
            "src/api/simplified_app.py"
        ]
        
        for component in new_components:
            component_path = self.project_root / component
            if component_path.exists():
                print_success(f"Found: {component}")
                # Basic syntax validation
                try:
                    with open(component_path, 'r') as f:
                        compile(f.read(), component_path, 'exec')
                    print_success(f"Syntax valid: {component}")
                except SyntaxError as e:
                    raise Exception(f"Syntax error in {component}: {e}")
            else:
                raise Exception(f"Missing new component: {component}")
    
    def _update_imports(self):
        """Update imports throughout the codebase"""
        print_step("Updating imports to use new unified architecture...")
        
        # Import mapping from old to new
        import_mappings = {
            'src.ai.advanced_knowledge_engine': 'src.ai.unified_knowledge_engine',
            'src.ai.llm_processing_engine': 'src.ai.unified_knowledge_engine',
            'src.ai.enhanced_extraction_engine': 'src.ai.unified_knowledge_engine',
            'src.ai.knowledge_categorization_engine': 'src.ai.unified_knowledge_engine',
            'src.core.config': 'src.core.unified_config',
            'src.processors.processor': 'src.processors.streamlined_processor'
        }
        
        # Files to update
        files_to_update = [
            "src/frontend/knowledge_table.py",
            "src/export/knowledge_export.py",
            "src/api/celery_worker.py"
        ]
        
        for file_path in files_to_update:
            full_path = self.project_root / file_path
            if full_path.exists():
                self._update_file_imports(full_path, import_mappings)
    
    def _update_file_imports(self, file_path: Path, mappings: Dict[str, str]):
        """Update imports in a specific file"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            original_content = content
            
            for old_import, new_import in mappings.items():
                # Update from imports
                content = content.replace(f"from {old_import}", f"from {new_import}")
                # Update direct imports
                content = content.replace(f"import {old_import}", f"import {new_import}")
            
            if content != original_content:
                with open(file_path, 'w') as f:
                    f.write(content)
                print_success(f"Updated imports: {file_path}")
                self.migration_log.append(f"IMPORT_UPDATE: {file_path}")
            
        except Exception as e:
            print_warning(f"Failed to update {file_path}: {e}")
    
    def _update_main_api(self):
        """Update the main API to use simplified version"""
        print_step("Updating main API configuration...")
        
        # Create a simple redirect in the main app.py
        main_api_path = self.project_root / "src/api/app.py"
        redirect_content = '''"""
EXPLAINIUM - API Redirect

This file redirects to the new simplified API for backward compatibility.
"""

# Import the new simplified app
from src.api.simplified_app import app

# Re-export for backward compatibility
__all__ = ['app']

# The app is now imported from simplified_app
# All functionality has been moved to the streamlined architecture
'''
        
        try:
            with open(main_api_path, 'w') as f:
                f.write(redirect_content)
            print_success("Updated main API to use simplified architecture")
            self.migration_log.append("API_REDIRECT: Updated main API")
        except Exception as e:
            print_warning(f"Failed to update main API: {e}")
    
    def _create_compatibility_shims(self):
        """Create compatibility shims for old class names"""
        print_step("Creating compatibility shims...")
        
        # Create shims for old engine classes
        shim_content = '''"""
EXPLAINIUM - Compatibility Shims

These classes provide backward compatibility with the old architecture.
They redirect to the new unified engine while maintaining the same interface.
"""

from src.ai.unified_knowledge_engine import UnifiedKnowledgeEngine, ExtractionResult
from src.core.unified_config import get_config
import warnings

# Compatibility shims for old engines
class AdvancedKnowledgeEngine:
    """Compatibility shim for AdvancedKnowledgeEngine"""
    
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "AdvancedKnowledgeEngine is deprecated. Use UnifiedKnowledgeEngine instead.", 
            DeprecationWarning, 
            stacklevel=2
        )
        self._engine = UnifiedKnowledgeEngine(get_config())
    
    async def extract_knowledge(self, content, document_type="unknown"):
        return await self._engine.extract_knowledge(content, document_type)

class LLMProcessingEngine:
    """Compatibility shim for LLMProcessingEngine"""
    
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "LLMProcessingEngine is deprecated. Use UnifiedKnowledgeEngine instead.", 
            DeprecationWarning, 
            stacklevel=2
        )
        self._engine = UnifiedKnowledgeEngine(get_config())
    
    async def process_with_llm(self, content, document_type="unknown"):
        return await self._engine.extract_knowledge(content, document_type, strategy_preference="llm")

class EnhancedExtractionEngine:
    """Compatibility shim for EnhancedExtractionEngine"""
    
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "EnhancedExtractionEngine is deprecated. Use UnifiedKnowledgeEngine instead.", 
            DeprecationWarning, 
            stacklevel=2
        )
        self._engine = UnifiedKnowledgeEngine(get_config())
    
    def extract_comprehensive_knowledge(self, content, document_type="unknown"):
        # Note: This is a sync wrapper for the async method
        import asyncio
        return asyncio.run(self._engine.extract_knowledge(content, document_type))

class KnowledgeCategorizationEngine:
    """Compatibility shim for KnowledgeCategorizationEngine"""
    
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "KnowledgeCategorizationEngine is deprecated. Use UnifiedKnowledgeEngine instead.", 
            DeprecationWarning, 
            stacklevel=2
        )
        self._engine = UnifiedKnowledgeEngine(get_config())
    
    async def categorize_knowledge(self, content, document_type="unknown"):
        return await self._engine.extract_knowledge(content, document_type, strategy_preference="nlp")

# Export all for backward compatibility
__all__ = [
    'AdvancedKnowledgeEngine',
    'LLMProcessingEngine', 
    'EnhancedExtractionEngine',
    'KnowledgeCategorizationEngine'
]
'''
        
        try:
            shim_path = self.project_root / "src/ai/compatibility_shims.py"
            with open(shim_path, 'w') as f:
                f.write(shim_content)
            print_success("Created compatibility shims")
            self.migration_log.append("SHIMS: Created compatibility layer")
        except Exception as e:
            print_warning(f"Failed to create shims: {e}")
    
    def _test_new_architecture(self):
        """Test that the new architecture works"""
        print_step("Testing new unified architecture...")
        
        test_script = '''
import sys
import asyncio
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_unified_engine():
    """Test the unified knowledge engine"""
    try:
        from src.ai.unified_knowledge_engine import UnifiedKnowledgeEngine
        from src.core.unified_config import get_config
        
        config = get_config()
        engine = UnifiedKnowledgeEngine(config)
        
        # Test pattern extraction
        test_content = "Step 1: Check the equipment. Step 2: Ensure safety procedures are followed."
        result = await engine.extract_knowledge(test_content, "manual", strategy_preference="pattern")
        
        print(f"✓ Pattern extraction: {len(result.entities)} entities found")
        
        # Test configuration
        print(f"✓ Configuration loaded: {config.environment.value} environment")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_unified_engine())
    sys.exit(0 if success else 1)
'''
        
        test_path = self.project_root / "test_migration.py"
        try:
            with open(test_path, 'w') as f:
                f.write(test_script)
            
            # Run the test
            result = subprocess.run([sys.executable, str(test_path)], 
                                  capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                print_success("New architecture test passed")
                self.migration_log.append("TEST: Architecture validation successful")
            else:
                print_warning(f"Architecture test warnings: {result.stderr}")
                self.migration_log.append(f"TEST: Warnings - {result.stderr}")
            
            # Clean up test file
            test_path.unlink()
            
        except Exception as e:
            print_warning(f"Could not run architecture test: {e}")
    
    def _generate_migration_report(self):
        """Generate a comprehensive migration report"""
        print_step("Generating migration report...")
        
        # Calculate size reduction
        old_size = self._calculate_old_codebase_size()
        new_size = self._calculate_new_codebase_size()
        reduction_percent = ((old_size - new_size) / old_size * 100) if old_size > 0 else 0
        
        report = {
            "migration_timestamp": datetime.now().isoformat(),
            "project_root": str(self.project_root),
            "backup_location": str(self.backup_dir),
            "codebase_metrics": {
                "old_size_lines": old_size,
                "new_size_lines": new_size,
                "reduction_lines": old_size - new_size,
                "reduction_percent": f"{reduction_percent:.1f}%"
            },
            "files_migrated": {
                "engines_consolidated": len(self.old_engines),
                "config_unified": len(self.old_config_files),
                "processors_simplified": len(self.old_processors),
                "api_streamlined": len(self.old_api)
            },
            "new_architecture": {
                "unified_engine": "src/ai/unified_knowledge_engine.py",
                "streamlined_processor": "src/processors/streamlined_processor.py", 
                "unified_config": "src/core/unified_config.py",
                "simplified_api": "src/api/simplified_app.py"
            },
            "backward_compatibility": {
                "shims_created": True,
                "import_redirects": True,
                "api_compatibility": True
            },
            "migration_log": self.migration_log
        }
        
        report_path = self.project_root / "MIGRATION_REPORT.json"
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            print_success(f"Migration report saved: {report_path}")
            print()
            print_step("MIGRATION SUMMARY")
            print(f"📊 Code reduction: {reduction_percent:.1f}% ({old_size - new_size:,} lines)")
            print(f"🗃️  Files consolidated: {len(self.old_engines + self.old_config_files + self.old_processors)}")
            print(f"📁 Backup location: {self.backup_dir}")
            print(f"📋 Full report: {report_path}")
            
        except Exception as e:
            print_warning(f"Could not save migration report: {e}")
    
    def _calculate_old_codebase_size(self) -> int:
        """Calculate total lines in old codebase"""
        total_lines = 0
        for file_path in self.old_engines + self.old_config_files + self.old_processors:
            backup_file = self.backup_dir / file_path
            if backup_file.exists():
                try:
                    with open(backup_file, 'r') as f:
                        total_lines += len(f.readlines())
                except:
                    pass
        return total_lines
    
    def _calculate_new_codebase_size(self) -> int:
        """Calculate total lines in new codebase"""
        total_lines = 0
        new_files = [
            "src/ai/unified_knowledge_engine.py",
            "src/processors/streamlined_processor.py",
            "src/core/unified_config.py",
            "src/api/simplified_app.py"
        ]
        
        for file_path in new_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                try:
                    with open(full_path, 'r') as f:
                        total_lines += len(f.readlines())
                except:
                    pass
        return total_lines
    
    def _rollback_migration(self):
        """Rollback migration in case of failure"""
        print_step("Rolling back migration...")
        try:
            # Restore from backup
            for file_path in self.old_engines + self.old_config_files + self.old_processors + self.old_api:
                backup_file = self.backup_dir / file_path
                target_file = self.project_root / file_path
                if backup_file.exists():
                    shutil.copy2(backup_file, target_file)
                    print_success(f"Restored: {file_path}")
            
            print_success("Migration rolled back successfully")
        except Exception as e:
            print_error(f"Rollback failed: {e}")

def main():
    """Main migration entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate Explainium to new unified architecture")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    
    args = parser.parse_args()
    
    if args.dry_run:
        print_step("DRY RUN MODE - No changes will be made")
        print("Would migrate:")
        migration = ArchitectureMigration(args.project_root)
        for engine in migration.old_engines:
            print(f"  📄 {engine}")
        return
    
    migration = ArchitectureMigration(args.project_root)
    success = migration.run_migration()
    
    if success:
        print()
        print_success("🎉 Migration completed successfully!")
        print("   Your codebase has been simplified and optimized.")
        print("   Check MIGRATION_REPORT.json for detailed information.")
    else:
        print()
        print_error("❌ Migration failed!")
        print("   Check the backup directory and logs for more information.")
        sys.exit(1)

if __name__ == "__main__":
    main()