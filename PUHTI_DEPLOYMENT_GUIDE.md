# Puhti Deployment Guide - Explainium 2.0

## Overview
This guide provides step-by-step instructions for deploying Explainium 2.0 on CSC's Puhti supercomputing environment.

## Prerequisites
- CSC account with Puhti access
- Project allocation (project_2015237)
- Basic knowledge of SSH and Linux commands

## Step 1: Connect to Puhti

### Initial Connection
```bash
ssh rbhandar@puhti.csc.fi
```

### Request Compute Resources
```bash
srun --account=project_2015237 --time=4:00:00 --mem=8G --cpus-per-task=4 --pty bash
```

## Step 2: Clone Repository

### Navigate to Project Directory
```bash
cd /scratch/project_2015237/
```

### Clone Repository
```bash
git clone https://github.com/imaddde867/explainium-2.0.git
cd explainium-2.0
```

## Step 3: Set Up Environment

### Create Virtual Environment
```bash
python3 -m venv venv_scratch
source venv_scratch/bin/activate
```

### Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Step 4: Configure Environment Variables

### Set CSC-Specific Variables
```bash
export ENVIRONMENT=production
export PYTHONPATH="${PYTHONPATH}:/scratch/project_2015237/explainium-2.0"
export DATABASE_URL=sqlite:///./explainium.db
export UPLOAD_DIRECTORY=/projappl/project_2015237/explainium-2.0/uploaded_files
export MAX_FILE_SIZE_MB=500
export LOG_LEVEL=INFO
```

### Set HPC Compatibility Variables
```bash
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TORCH_DISABLE_DYNAMO=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export AI_ENGINE_TIMEOUT=10
```

## Step 5: Deploy Application

### Run Deployment Script
```bash
chmod +x deploy-puhti.sh
./deploy-puhti.sh
```

### Verify Deployment
The script will:
- Load required modules
- Activate virtual environment
- Set environment variables
- Initialize database
- Test essential components
- Start Streamlit application

## Step 6: Access Application

### Create SSH Tunnel (from local machine)
```bash
ssh -L 8501:r18c01.bullx:8501 -L 8000:r18c01.bullx:8000 rbhandar@puhti.csc.fi
```

### Access URLs
- **Streamlit Dashboard**: http://localhost:8501
- **API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## Step 7: Monitor Application

### Check Process Status
```bash
ps aux | grep python
ps aux | grep streamlit
```

### Monitor System Resources
```bash
top
htop
```

### Check Logs
```bash
tail -f logs/app.log
```

## Troubleshooting

### Common Issues

#### 1. Module Loading Errors
```bash
# Solution: Skip module loading
# Comment out module load commands in deploy-puhti.sh
```

#### 2. AI Engine Initialization Timeout
```bash
# Solution: Increase timeout or disable AI engines
export AI_ENGINE_TIMEOUT=30
```

#### 3. Memory Issues
```bash
# Solution: Reduce memory usage
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

#### 4. Network Connectivity
```bash
# Solution: Use offline mode
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
```

### Performance Optimization

#### 1. Resource Allocation
- Request appropriate CPU and memory resources
- Use `--cpus-per-task=4` for optimal performance
- Allocate at least 8GB RAM for AI processing

#### 2. Environment Variables
- Set `OMP_NUM_THREADS=1` to prevent thread conflicts
- Use offline mode for AI models
- Enable HPC-specific optimizations

#### 3. Application Settings
- Use SQLite for database (no PostgreSQL setup required)
- Enable file watching for development
- Set appropriate timeout values

## Security Considerations

### Environment Variables
- Never commit sensitive data to repository
- Use environment variables for configuration
- Set appropriate file permissions

### File Access
- Use project-specific directories
- Set proper upload directory permissions
- Limit file size uploads

## Maintenance

### Regular Updates
```bash
git pull origin main
./deploy-puhti.sh
```

### Cleanup
```bash
# Remove old log files
rm -rf logs/*.log

# Clear cache
rm -rf /tmp/transformers_cache
rm -rf /tmp/hf_home
```

### Backup
```bash
# Backup database
cp explainium.db explainium.db.backup

# Backup uploaded files
tar -czf uploaded_files_backup.tar.gz uploaded_files/
```

## Support

### CSC Documentation
- [Puhti User Guide](https://docs.csc.fi/computing/systems-puhti/)
- [CSC Support](https://docs.csc.fi/support/)

### Project Issues
- Check GitHub issues
- Contact project maintainer
- Review deployment logs

## Conclusion

This deployment guide provides comprehensive instructions for running Explainium 2.0 on Puhti. The application is optimized for HPC environments and includes fallback mechanisms for compatibility issues.

For additional support or questions, refer to the project documentation or contact the development team.
