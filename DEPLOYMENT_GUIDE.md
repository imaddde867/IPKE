# 🚀 EXPLAINIUM Deployment Guide

## Quick Start (Recommended)

### 1. Prerequisites
- Docker and Docker Compose installed
- 4GB+ RAM available
- 10GB+ disk space

### 2. Clone and Start
```bash
git clone https://github.com/imaddde867/explainium-2.0
cd explainium-2.0
docker-compose up --build -d
```

### 3. Initialize Database
```bash
docker-compose exec app alembic upgrade head
```

### 4. Access Application
- **Web Interface**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Production Deployment

### 1. Environment Configuration
```bash
# Copy and customize environment file
cp .env.example .env

# Edit .env file with production values
nano .env
```

**Key production changes:**
```bash
ENVIRONMENT=production
DEBUG=false
DB_PASSWORD=your_secure_password
CORS_ORIGINS=https://yourdomain.com
LOG_LEVEL=WARNING
```

### 2. Security Considerations
- Change default database password
- Configure proper CORS origins
- Use HTTPS in production
- Set up proper firewall rules
- Regular security updates

### 3. Performance Tuning
```bash
# Increase worker concurrency
CELERY_CONCURRENCY=8

# Optimize database connections
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=40

# Adjust file size limits
MAX_FILE_SIZE_MB=200
```

### 4. Monitoring
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

## Scaling

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

## Support

- **Documentation**: [README.md](README.md)
- **Health Monitoring**: http://localhost:8000/health/detailed