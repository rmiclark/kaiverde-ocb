#!/bin/bash
# deploy.sh - Google Cloud Run deployment script

# Set variables - CHANGE THESE TO MATCH YOUR PROJECT
PROJECT_ID="kaiverde"  # Replace with your actual project ID
SERVICE_NAME="crystal-ball-api"
REGION="us-central1"  # Change if you prefer a different region
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"

echo "🚀 Deploying Crystal Ball API to Google Cloud Run"

# Check if PROJECT_ID is set
if [ "$PROJECT_ID" = "kaiverde" ]; then
    echo "❌ Please update PROJECT_ID in this script with your actual GCP project ID"
    echo "   You can find your project ID in the Google Cloud Console"
    exit 1
fi

# Authenticate (if not already done)
echo "📋 Checking authentication..."
gcloud auth list

# Set project
echo "🏗️ Setting project: $PROJECT_ID"
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "🔧 Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build and push Docker image
echo "📦 Building Docker image..."
gcloud builds submit --tag $IMAGE_NAME .

# Deploy to Cloud Run
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image $IMAGE_NAME \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 3600 \
  --max-instances 10 \
  --concurrency 10 \
  --set-env-vars PORT=8080

# Get the service URL
echo "✅ Deployment complete!"
echo ""
echo "🌐 Your service URL is:"
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)')
echo "$SERVICE_URL"
echo ""
echo "🧪 Test your API:"
echo "curl $SERVICE_URL/health"
echo ""
echo "📊 Example API call:"
echo "curl $SERVICE_URL/example"
echo ""
echo "📝 Update your Google Sheets script with this URL:"
echo "const API_URL = '$SERVICE_URL';"