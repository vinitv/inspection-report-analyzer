#!/bin/bash

# Property Inspection Analyzer - Cloud Run Deployment Script

# Configuration
PROJECT_ID="aim-07"  # Replace with your Google Cloud project ID
SERVICE_NAME="inspection-analyzer-beta"
REGION="us-central1"
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"

# Load environment variables from .env file
if [ -f .env ]; then
    echo "📄 Loading environment variables from .env file..."
    # More robust way to load .env file
    while IFS= read -r line; do
        # Skip empty lines and comments
        if [[ -n "$line" && ! "$line" =~ ^[[:space:]]*# ]]; then
            # Export the variable
            export "$line"
            echo "   Loaded: ${line%%=*}"
        fi
    done < .env
else
    echo "⚠️  .env file not found. Please create one with your API keys."
    echo "Example .env file:"
    echo "OPENAI_API_KEY=your-openai-key"
    echo "REPAIR_API_KEY=your-repair-api-key"
    echo "LANGSMITH_API_KEY=your-langsmith-key"
    exit 1
fi

# Check if required API keys are set
echo "🔍 Checking environment variables..."

if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ OPENAI_API_KEY not found in .env file"
    echo "   Current value: '$OPENAI_API_KEY'"
    exit 1
else
    echo "✅ OPENAI_API_KEY found (length: ${#OPENAI_API_KEY})"
fi

if [ -z "$REPAIR_API_KEY" ]; then
    echo "❌ REPAIR_API_KEY not found in .env file"
    echo "   Current value: '$REPAIR_API_KEY'"
    exit 1
else
    echo "✅ REPAIR_API_KEY found (length: ${#REPAIR_API_KEY})"
fi

if [ -z "$LANGSMITH_API_KEY" ]; then
    echo "⚠️  LANGSMITH_API_KEY not found in .env file (optional)"
else
    echo "✅ LANGSMITH_API_KEY found (length: ${#LANGSMITH_API_KEY})"
fi

if [ -z "$TAVILY_API_KEY" ]; then
    echo "⚠️  TAVILY_API_KEY not found in .env file (optional)"
else
    echo "✅ TAVILY_API_KEY found (length: ${#TAVILY_API_KEY})"
fi

echo "✅ Environment variables loaded successfully"

echo "🚀 Deploying Property Inspection Analyzer to Cloud Run..."

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI is not installed. Please install it first."
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install it first."
    exit 1
fi

# Set the project
echo "📋 Setting project to $PROJECT_ID..."
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "🔧 Enabling required APIs..."
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build and push the Docker image
echo "🏗️ Building Docker image..."
docker build --platform linux/amd64 -t $IMAGE_NAME .

echo "📤 Pushing image to Container Registry..."
docker push $IMAGE_NAME

# Deploy to Cloud Run
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image $IMAGE_NAME \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --concurrency 80 \
    --max-instances 10 \
    --set-env-vars "OPENAI_API_KEY=$OPENAI_API_KEY" \
    --set-env-vars "REPAIR_API_KEY=$REPAIR_API_KEY" \
    --set-env-vars "LANGSMITH_API_KEY=$LANGSMITH_API_KEY" \
    --set-env-vars "TAVILY_API_KEY=$TAVILY_API_KEY" \
    --set-env-vars "COHERE_API_KEY=$COHERE_API_KEY" \
    --set-env-vars "LANGCHAIN_TRACING_V2=$LANGCHAIN_TRACING_V2" \
    --set-env-vars "LANGSMITH_ENDPOINT=$LANGSMITH_ENDPOINT" \
    --set-env-vars "LANGSMITH_PROJECT=$LANGSMITH_PROJECT"

echo "✅ Deployment complete!"
echo "🌐 Your app is available at:"
gcloud run services describe $SERVICE_NAME --region $REGION --format="value(status.url)"

echo ""
echo "📝 Next steps:"
echo "1. Create a .env file with your API keys:"
echo "   OPENAI_API_KEY=your-openai-key"
echo "   REPAIR_API_KEY=your-repair-api-key"
echo "   LANGSMITH_API_KEY=your-langsmith-key"
echo "2. Test the application"
echo "3. Set up a custom domain (optional)"
