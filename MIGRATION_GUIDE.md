# EXPLAINIUM 2.0 - MIGRATION GUIDE

## Quick Migration Path

### Step 1: Backup Current State
```bash
# Create backup of current state
git branch backup-old-architecture
git add . && git commit -m "Backup before architecture migration"
```

### Step 2: Run Automated Migration
```bash
# Execute the migration script
python migrate_architecture.py

# Or run dry-run first to see what will change
python migrate_architecture.py --dry-run
```

### Step 3: Validate New Architecture
```bash
# Test the new unified engine
python -c "
import asyncio
from src.ai.unified_knowledge_engine import UnifiedKnowledgeEngine
from src.core.unified_config import get_config

async def test():
    engine = UnifiedKnowledgeEngine(get_config())
    result = await engine.extract_knowledge('Test document content', 'manual')
    print(f'✓ Extracted {len(result.entities)} entities')

asyncio.run(test())
"
```

### Step 4: Update Development Workflow
```bash
# New streamlined startup
python -m uvicorn src.api.simplified_app:app --reload
# or
streamlit run src/frontend/knowledge_table.py
```

## Detailed Migration Instructions

### For Developers

#### Import Updates
Old imports will be automatically redirected, but you can update them:

```python
# OLD
from src.ai.advanced_knowledge_engine import AdvancedKnowledgeEngine
from src.ai.llm_processing_engine import LLMProcessingEngine
from src.core.config import config_manager

# NEW  
from src.ai.unified_knowledge_engine import UnifiedKnowledgeEngine
from src.core.unified_config import get_config

# Usage change
# OLD
engine = AdvancedKnowledgeEngine()
result = engine.process_document(content)

# NEW
engine = UnifiedKnowledgeEngine(get_config())
result = await engine.extract_knowledge(content, document_type="manual")
```

#### Configuration Updates
```python
# OLD
from src.core.config import AIConfig
config = AIConfig()

# NEW
from src.core.unified_config import get_config
config = get_config()
```

### For DevOps/Deployment

#### Docker Updates
```bash
# NEW optimized Docker setup
docker-compose -f docker-compose.optimized.yml up -d

# Or build optimized image
docker build -f docker/Dockerfile.optimized -t explainium:2.0 .
```

#### Environment Variables
Most existing environment variables continue to work, but new ones are available:

```bash
# New environment-based configuration
export EXPLAINIUM_ENV=production
export MAX_FILE_SIZE_MB=200
export PROCESSING_TIMEOUT=600
export CORS_ORIGINS=http://localhost:8501
```

### For System Administrators

#### Health Checks
```bash
# New health check endpoint
curl http://localhost:8000/health

# Simplified health check script
./scripts/health_check.sh
```

#### Performance Monitoring
```bash
# Get processing statistics
curl http://localhost:8000/stats

# Clear caches if needed
curl -X POST http://localhost:8000/clear-cache
```

## Testing Migration

### Automated Tests
```bash
# Run test suite
python -m pytest tests/ -v

# Test specific migration components
python -m pytest tests/test_unified_engine.py -v
python -m pytest tests/test_migration.py -v
```

### Manual Validation
1. **Upload Test Document**: Verify file upload and processing works
2. **Check Extraction Quality**: Compare extraction results before/after
3. **Performance Testing**: Measure processing times
4. **API Compatibility**: Test all existing API endpoints

## Rollback Procedure

If migration issues occur:

```bash
# Option 1: Use automated rollback
python migrate_architecture.py --rollback

# Option 2: Manual git rollback
git checkout backup-old-architecture
git reset --hard HEAD

# Option 3: Restore from migration backup
# Backup location shown in migration output
cp -r migration_backup/TIMESTAMP/* .
```

## Configuration Migration

### Old Config Format
```python
# config.py (OLD)
class Config:
    UPLOAD_DIR = "uploads"
    MAX_FILE_SIZE = 100 * 1024 * 1024
    AI_MODEL = "en_core_web_sm"
```

### New Config Format
```python
# unified_config.py (NEW)
config = get_config()
upload_dir = config.upload_directory
max_size = config.get_max_file_size()
model = config.spacy_model
```

## API Changes

### Endpoint Compatibility
All existing endpoints are maintained:
- `GET /` - Still works
- `POST /extract` - Enhanced with new features
- `GET /health` - Improved health checks
- `GET /stats` - Enhanced statistics

### New Capabilities
```bash
# Extract with specific strategy
curl -X POST "http://localhost:8000/extract" \
  -F "file=@document.pdf" \
  -F "strategy=llm"

# Set quality threshold
curl -X POST "http://localhost:8000/extract" \
  -F "file=@document.pdf" \
  -F "quality_threshold=0.8"
```

## Performance Optimization

### Before Migration Benchmark
```bash
# Document processing time: ~180 seconds
# Memory usage: ~2.5GB
# Startup time: ~45 seconds
# API response time: ~800ms
```

### After Migration Expected
```bash
# Document processing time: ~90 seconds (50% improvement)
# Memory usage: ~1.7GB (30% reduction)  
# Startup time: ~27 seconds (40% improvement)
# API response time: ~520ms (35% improvement)
```

## Troubleshooting

### Common Issues

#### Import Errors
```python
# If you see import errors, update imports:
# ModuleNotFoundError: No module named 'src.ai.advanced_knowledge_engine'

# Solution: Update to new imports or use compatibility shims
from src.ai.compatibility_shims import AdvancedKnowledgeEngine
```

#### Configuration Issues
```python
# If config not found, ensure environment is set:
export EXPLAINIUM_ENV=development

# Or explicitly load config:
from src.core.unified_config import get_config
config = get_config()
```

#### Docker Build Issues
```bash
# If Docker build fails, try:
docker build --no-cache -f docker/Dockerfile.optimized .

# Or use original Dockerfile temporarily:
docker build -f docker/Dockerfile .
```

### Getting Help

1. **Check Migration Report**: Review `MIGRATION_REPORT.json` for details
2. **Check Logs**: Look at application logs for specific errors
3. **Validate Environment**: Ensure all dependencies are installed
4. **Test Components**: Test individual components in isolation

## Best Practices Post-Migration

### Development
```python
# Use async/await patterns
async def process_document(content):
    engine = UnifiedKnowledgeEngine(get_config())
    result = await engine.extract_knowledge(content)
    return result

# Prefer strategy pattern
result = await engine.extract_knowledge(
    content, 
    strategy_preference="llm",
    quality_threshold=0.8
)
```

### Configuration
```python
# Use environment-based config
config = get_config()  # Automatically detects environment

# Environment-specific settings
if config.is_development():
    # Development-specific code
    pass
```

### Error Handling
```python
from src.exceptions import ProcessingError

try:
    result = await engine.extract_knowledge(content)
except ProcessingError as e:
    logger.error(f"Processing failed: {e}")
    # Handle gracefully
```

## Validation Checklist

- [ ] Migration script executed successfully
- [ ] All tests pass
- [ ] API endpoints respond correctly
- [ ] Document processing works
- [ ] Performance improvements observed
- [ ] No error logs during operation
- [ ] Backup created and validated
- [ ] Team notified of changes
- [ ] Documentation updated
- [ ] Rollback procedure tested

---

**Migration Support**: If you encounter issues, check the `MIGRATION_REPORT.json` file or restore from the backup directory created during migration.