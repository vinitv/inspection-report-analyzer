# 🚀 Cloud Run Deployment Guide

This guide will help you deploy the Property Inspection Analyzer to Google Cloud Run.

## 📋 Prerequisites

1. **Google Cloud Account** with billing enabled
2. **Google Cloud CLI** installed and configured
3. **Docker** installed locally
4. **API Keys** for OpenAI and Repair Cost API

## 🔧 Setup Steps

### 1. Create a Google Cloud Project

```bash
# Create a new project (or use existing)
gcloud projects create your-project-id --name="Inspection Analyzer"

# Set the project
gcloud config set project your-project-id
```

### 2. Enable Required APIs

```bash
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your-openai-api-key
REPAIR_API_KEY=your-repair-api-key
LANGSMITH_API_KEY=your-langsmith-key
```

### 4. Update Deployment Configuration

Edit `deploy.sh` and replace:
- `your-project-id` with your actual Google Cloud project ID
- `your-openai-key` with your OpenAI API key
- `your-repair-api-key` with your repair API key

## 🚀 Deployment Options

### Option 1: Manual Deployment

```bash
# Make the script executable
chmod +x deploy.sh

# Run the deployment
./deploy.sh
```

### Option 2: Cloud Build (Recommended)

```bash
# Submit build to Cloud Build
gcloud builds submit --config cloudbuild.yaml
```

### Option 3: Direct gcloud Commands

```bash
# Build and push image
docker build -t gcr.io/your-project-id/inspection-analyzer .
docker push gcr.io/your-project-id/inspection-analyzer

# Deploy to Cloud Run
gcloud run deploy inspection-analyzer \
    --image gcr.io/your-project-id/inspection-analyzer \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --concurrency 80 \
    --max-instances 10 \
    --set-env-vars "OPENAI_API_KEY=your-key" \
    --set-env-vars "REPAIR_API_KEY=your-key"
```

## 🔍 Configuration Details

### Resource Allocation
- **Memory**: 2GB (sufficient for AI processing)
- **CPU**: 2 vCPUs (good performance for concurrent users)
- **Timeout**: 300 seconds (5 minutes for long AI operations)
- **Concurrency**: 80 requests per instance
- **Max Instances**: 10 (scales based on demand)

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key
- `REPAIR_API_KEY`: Your repair cost API key
- `PORT`: Automatically set by Cloud Run (8080)
- `HOST`: Automatically set by Cloud Run (0.0.0.0)

## 🌐 Accessing Your App

After deployment, your app will be available at:
```
https://inspection-analyzer-xxxxx-uc.a.run.app
```

## 📊 Monitoring

### View Logs
```bash
gcloud logs tail --service=inspection-analyzer
```

### Check Service Status
```bash
gcloud run services describe inspection-analyzer --region us-central1
```

### Monitor Performance
- Visit Google Cloud Console > Cloud Run
- Check metrics, logs, and performance

## 🔄 Continuous Deployment

### GitHub Actions (Optional)

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Cloud Run

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Deploy to Cloud Run
      uses: google-github-actions/deploy-cloudrun@v0
      with:
        service: inspection-analyzer
        image: gcr.io/${{ secrets.GCP_PROJECT_ID }}/inspection-analyzer
        region: us-central1
        env_vars: |
          OPENAI_API_KEY=${{ secrets.OPENAI_API_KEY }}
          REPAIR_API_KEY=${{ secrets.REPAIR_API_KEY }}
```

## 🛠️ Troubleshooting

### Common Issues

1. **Build Fails**: Check Dockerfile and requirements.txt
2. **Runtime Errors**: Check logs with `gcloud logs tail`
3. **API Errors**: Verify environment variables are set correctly
4. **Memory Issues**: Increase memory allocation if needed

### Debug Commands

```bash
# Check service logs
gcloud logs tail --service=inspection-analyzer

# Test locally
docker run -p 8080:8080 gcr.io/your-project-id/inspection-analyzer

# Check service configuration
gcloud run services describe inspection-analyzer --region us-central1
```

## 💰 Cost Optimization

- **Min Instances**: 0 (scales to zero when not in use)
- **Max Instances**: 10 (prevents runaway costs)
- **Memory**: 2GB (optimized for performance/cost ratio)
- **Region**: us-central1 (good balance of cost and latency)

## 🔒 Security Considerations

- **HTTPS**: Automatically enabled by Cloud Run
- **Authentication**: Currently set to allow unauthenticated access
- **API Keys**: Stored as environment variables (consider Secret Manager for production)
- **CORS**: Configured for web access

## 📈 Scaling

The service automatically scales based on:
- Number of incoming requests
- CPU and memory usage
- Response times

No manual scaling configuration needed!
