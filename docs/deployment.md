# Deployment Guide

## Prerequisites

- Docker Desktop (includes Docker Engine and Docker Compose)
- 8GB+ RAM recommended for AI models
- Stable internet connection for initial model downloads

## Quick Deployment

### Local Development
```bash
# Clone and start
git clone https://github.com/imaddde867/explainium-2.0
cd explainium-2.0
docker-compose up --build -d

# Initialize database
docker-compose exec app alembic upgrade head

# Verify services
docker-compose ps
```

### Production Deployment
```bash
# Use production configuration
docker-compose -f docker-compose.prod.yml up --build -d

# Initialize with production settings
docker-compose -f docker-compose.prod.yml exec app alembic upgrade head
```

## Service Architecture

**Core Services:**
- `app`: FastAPI application (port 8000)
- `celery_worker`: Background task processing
- `db`: PostgreSQL database (port 5432)
- `redis`: Task queue and caching (port 6379)
- `elasticsearch`: Search engine (port 9200)
- `tika`: Document processing (port 9998)

**Frontend (Optional):**
```bash
cd src/frontend
npm install && npm start  # Port 3000
```

## Configuration

### Environment Variables
```bash
# Database
DATABASE_URL=postgresql://user:password@db:5432/explainium

# Redis
REDIS_URL=redis://redis:6379/0

# Elasticsearch
ELASTICSEARCH_URL=http://elasticsearch:9200

# Tika
TIKA_SERVER_URL=http://tika:9998
```

### Resource Allocation
```yaml
# docker-compose.yml
services:
  app:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
  
  celery_worker:
    deploy:
      resources:
        limits:
          memory: 4G  # AI models require more memory
          cpus: '2.0'
```

## Monitoring and Maintenance

### Health Checks
```bash
# Service status
docker-compose ps

# Application health
curl http://localhost:8000/health

# Detailed health check
curl http://localhost:8000/health/detailed
```

### Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f app
docker-compose logs -f celery_worker
```

### Database Maintenance
```bash
# Backup
docker-compose exec db pg_dump -U postgres explainium > backup.sql

# Restore
docker-compose exec -T db psql -U postgres explainium < backup.sql

# Reset (development only)
docker-compose down -v
docker-compose up --build -d
```

## Scaling

### Horizontal Scaling
```yaml
# docker-compose.yml
services:
  celery_worker:
    deploy:
      replicas: 3  # Multiple workers for parallel processing
```

### Load Balancing
```yaml
services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
```

## Troubleshooting

### Common Issues

**Port Conflicts:**
```bash
# Check port usage
netstat -tulpn | grep :8000

# Modify ports in docker-compose.yml if needed
```

**Memory Issues:**
```bash
# Increase Docker memory allocation
# Docker Desktop > Settings > Resources > Memory

# Monitor container memory usage
docker stats
```

**Model Download Failures:**
```bash
# Check internet connectivity
# Restart celery worker
docker-compose restart celery_worker
```

**Database Connection Issues:**
```bash
# Check database status
docker-compose exec db pg_isready -U postgres

# Reset database connection
docker-compose restart app
```

### Performance Optimization

**Database:**
- Regular VACUUM and ANALYZE operations
- Monitor query performance with pg_stat_statements
- Consider connection pooling for high-load scenarios

**Elasticsearch:**
- Adjust heap size based on available memory
- Monitor cluster health and index performance
- Configure appropriate refresh intervals

**AI Models:**
- Cache model downloads in persistent volumes
- Use GPU acceleration if available
- Consider model quantization for memory efficiency

## Security Considerations

**Production Checklist:**
- [ ] Change default passwords
- [ ] Enable SSL/TLS encryption
- [ ] Configure firewall rules
- [ ] Implement authentication and authorization
- [ ] Regular security updates
- [ ] Monitor access logs
- [ ] Backup encryption