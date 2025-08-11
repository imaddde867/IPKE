# Explainium Deployment Guide

## Quick Start (Recommended)

### 1. Prerequisites
- Docker and Docker Compose installed
- 4GB+ RAM available (8GB+ recommended)
- 10GB+ disk space

### 2. One-Command Deployment
```bash
git clone https://github.com/imaddde867/explainium-2.0
cd explainium-2.0

# Production deployment
./deploy.sh

# Or development deployment
./deploy.sh --dev

# Or quick deployment (skip health checks)
./deploy.sh --quick
```

### 3. Alternative: Using Make
```bash
# Production
make prod

# Development
make dev

# Quick start
make quick
```

### 4. Access Application
- Web Interface: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## Production Deployment

### 1. Environment Configuration
The deployment script automatically creates a `.env` file with production settings. You can customize it:

```bash
# Edit environment configuration
nano .env
```

**Key production settings:**
```bash
ENVIRONMENT=production
API_DEBUG=false
DB_PASSWORD=your_secure_password
CORS_ORIGINS=https://yourdomain.com
LOG_LEVEL=INFO
SECRET_KEY=your-32-character-secret-key
ENABLE_HTTPS=true
```

### 2. Security Considerations
- Deployment script generates a random SECRET_KEY (review/replace as needed)
- Change default database password in production
- Configure strict CORS origins for your domain
- Enable HTTPS (reverse proxy or load balancer)
- Use strong unique passwords for all services

### 3. Architecture Highlights
- Single API service: `src/api/app.py`
- Processing orchestration: `src/processors/processor.py`
- Central configuration: `src/core/config.py`
- Database layer: `src/database/`
- Automated deployment with health checks
- Apply firewall rules and schedule security patching

### 4. Performance Tuning
```bash
# Increase worker concurrency
CELERY_CONCURRENCY=8

# Optimize database connections
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=40

# Adjust file size limits
MAX_FILE_SIZE_MB=200
```

### 5. Monitoring
```bash
# Check service status
docker-compose ps

# View logs
docker-compose logs app --tail=100
docker-compose logs celery_worker --tail=100

# Monitor resource usage
docker stats
```

## Troubleshooting

### Common Issues

**Services not starting:**
```bash
docker-compose down
docker-compose up --build -d
```

**Database connection errors:**
```bash
docker-compose restart db
docker-compose logs db
```

**Processing stuck:**
```bash
docker-compose restart celery_worker
```

**Out of disk space:**
```bash
# Clean up Docker
docker system prune -a

# Clean up uploaded files
rm -rf uploaded_files/*
```

### Health Checks
```bash
# Basic health
curl http://localhost:8000/health

# Detailed health
curl http://localhost:8000/health/detailed

# Check specific services
curl http://localhost:9200/_cluster/health  # Elasticsearch
curl http://localhost:9998/tika             # Tika
```

## Backup and Recovery

### Database Backup
```bash
# Create backup
docker-compose exec db pg_dump -U user knowledge_db > backup.sql

# Restore backup
docker-compose exec -T db psql -U user knowledge_db < backup.sql
```

### Full System Backup
```bash
# Stop services
docker-compose down

# Backup data volumes
docker run --rm -v explainium-20_postgres_data:/data -v $(pwd):/backup alpine tar czf /backup/postgres_backup.tar.gz -C /data .
docker run --rm -v explainium-20_elasticsearch_data:/data -v $(pwd):/backup alpine tar czf /backup/elasticsearch_backup.tar.gz -C /data .

# Restart services
docker-compose up -d
```

## Scaling & Resources

### Horizontal Scaling
```bash
# Scale Celery workers
docker-compose up --scale celery_worker=4 -d

# Scale API instances (requires load balancer)
docker-compose up --scale app=3 -d
```

### Resource Limits
```yaml
# Add to docker-compose.yml
services:
  app:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
```

## Updates

### Application Updates
```bash
# Pull latest changes
git pull origin master

# Rebuild and restart
docker-compose up --build -d

# Run any new migrations
docker-compose exec app alembic upgrade head
```

### Docker Updates
```bash
# Update base images
docker-compose pull
docker-compose up -d
```

## Support & References

- Documentation: [README.md](README.md)
- Technical Specs: [TECHNICAL_SPECS.md](TECHNICAL_SPECS.md)
- Detailed Health: http://localhost:8000/health/detailed