# #!/bin/bash
# # Emergency Fix for Cloud Run Deployment Timeout
# # This script will fix the deployment by excluding unnecessary files

# echo "🚨 Emergency Fix: Stopping Virtual Environment Upload"
# echo "=================================================="

# # Step 1: Create or overwrite .gcloudignore
# cat > .gcloudignore <<'EOF'
# # Virtual Environments (CRITICAL - DO NOT UPLOAD)
# nvenv/
# venv/
# env/
# fenv/
# __pycache__/

# # Data files
# *.csv
# *.xlsx
# *.pkl
# *.joblib
# data/
# models/

# # Development files
# *.log
# *.db
# .DS_Store
# *.md
# .git
# .gitignore
# EOF

# echo "✅ Created .gcloudignore"

# # Step 2: Force Python 3.10
# echo "python-3.10" > runtime.txt
# echo "✅ Created runtime.txt"

# # Step 3: Create minimal requirements.txt (NO PYTORCH)
# cat > requirements.txt <<'EOF'
# # FastAPI Core
# fastapi==0.109.0
# uvicorn[standard]==0.27.0
# python-multipart==0.0.6
# pydantic==2.5.3

# # Google Cloud
# google-cloud-storage==2.14.0
# google-auth==2.27.0

# # Data Science (Python 3.10 compatible, NO PYTORCH)
# pandas==2.2.0
# numpy==1.26.4
# scikit-learn==1.4.0

# # Security
# python-jose[cryptography]==3.3.0
# passlib[bcrypt]==1.7.4
# python-dateutil==2.8.2
# EOF

# echo "✅ Created minimal requirements.txt (NO PyTorch)"

# # Step 4: Verify what will be uploaded
# echo ""
# echo "📦 Checking files to be uploaded..."
# gcloud meta list-files-for-upload | head -20

# # Step 5: Deploy with proper settings
# echo ""
# echo "🚀 Deploying to Cloud Run..."
# gcloud run deploy financial-stress-api \
#   --source . \
#   --region us-east1 \
#   --allow-unauthenticated \
#   --memory 2Gi \
#   --cpu 2 \
#   --timeout 900 \
#   --max-instances 10 \
#   --set-env-vars DATA_BUCKET=mlops-financial-stress-data,ENVIRONMENT=production

# # Step 6: Get the URL
# echo ""
# echo "🎉 Getting service URL..."
# API_URL=$(gcloud run services describe financial-stress-api \
#   --region us-east1 \
#   --format 'value(status.url)')

# echo ""
# echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# echo "✅ Deployment Complete!"
# echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# echo ""
# echo "📡 API URL:        ${API_URL}"
# echo "❤️  Health Check:  ${API_URL}/health"
# echo "📚 API Docs:       ${API_URL}/docs"
# echo ""
# echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"



#!/bin/bash
# Optimized Cloud Run Deployment Script
# Fixes timeout issues and missing dependencies

echo "🚀 Financial Stress Test Platform - Cloud Run Deployment"
echo "=========================================================="

# Step 1: Create or overwrite .gcloudignore
cat > .gcloudignore <<'EOF'
# Virtual Environments (CRITICAL - DO NOT UPLOAD)
nvenv/
venv/
env/
fenv/
.venv/
__pycache__/

# Data files (HUGE - never upload)
*.csv
*.xlsx
*.pkl
*.joblib
*.h5
*.pt
*.pth
data/
models/
outputs/
datasets/

# Development files
*.log
*.db
*.sqlite
*.sqlite3
.DS_Store
*.md
README*
.git/
.gitignore

# IDE
.vscode/
.idea/
*.swp
*.swo

# Testing
tests/
test_*
*_test.py

# Backup files
*_backup.py
*.bak
*.tmp
EOF

echo "✅ Created .gcloudignore"

# Step 2: Force Python 3.11 (better support than 3.10)
echo "python-3.11" > runtime.txt
echo "✅ Created runtime.txt with Python 3.11"

# Step 3: Create optimized requirements.txt
cat > requirements.txt <<'EOF'
# Core FastAPI (minimal versions for speed)
fastapi==0.109.0
uvicorn==0.27.0
python-multipart==0.0.6
pydantic==2.5.3

# Google Cloud
google-cloud-storage==2.14.0

# ML Libraries (binary wheels - fast install)
pandas==2.2.0
numpy==1.26.4
scikit-learn==1.4.0
lightgbm==4.1.0
shap==0.44.0

# Security
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
bcrypt==4.1.2
PyJWT==2.8.0
EOF

echo "✅ Created optimized requirements.txt (with lightgbm + shap)"

# Step 4: Verify what will be uploaded
echo ""
echo "📦 Checking files to be uploaded..."
FILE_COUNT=$(gcloud meta list-files-for-upload 2>/dev/null | wc -l)
echo "   Total files: $FILE_COUNT"
echo ""
echo "   First 20 files:"
gcloud meta list-files-for-upload 2>/dev/null | head -20

# Step 5: Deploy with INCREASED resources (your models need more memory!)
echo ""
echo "🚀 Deploying to Cloud Run with optimized settings..."
echo ""
gcloud run deploy financial-stress-api \
  --source . \
  --region us-east1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 900 \
  --max-instances 10 \
  --min-instances 0 \
  --concurrency 80 \
  --set-env-vars DATA_BUCKET=mlops-financial-stress-data,ENVIRONMENT=production \
  --platform managed

# Check if deployment succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ DEPLOYMENT SUCCESSFUL!"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Step 6: Get the URL
    echo ""
    echo "🎉 Getting service details..."
    API_URL=$(gcloud run services describe financial-stress-api \
      --region us-east1 \
      --format 'value(status.url)' 2>/dev/null)
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🌐 Service Information"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "📡 API URL:        ${API_URL}"
    echo "❤️  Health Check:  ${API_URL}/health"
    echo "📚 API Docs:       ${API_URL}/docs"
    echo "🔐 Login:          ${API_URL}/api/v1/auth/login"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🧪 Quick Tests:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "# Test health endpoint:"
    echo "curl ${API_URL}/health"
    echo ""
    echo "# Test API health:"
    echo "curl ${API_URL}/api/v1/health"
    echo ""
    echo "# Login (default user):"
    echo "curl -X POST ${API_URL}/api/v1/auth/login \\"
    echo "  -H 'Content-Type: application/json' \\"
    echo "  -d '{\"username\":\"admin\",\"password\":\"admin123\"}'"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Test the health endpoint
    echo ""
    echo "🧪 Testing health endpoint..."
    HEALTH_RESPONSE=$(curl -s -w "\n%{http_code}" "${API_URL}/health" 2>/dev/null)
    HTTP_CODE=$(echo "$HEALTH_RESPONSE" | tail -n 1)
    
    if [ "$HTTP_CODE" = "200" ]; then
        echo "✅ Health check passed!"
        echo "$HEALTH_RESPONSE" | head -n -1 | jq '.' 2>/dev/null || echo "$HEALTH_RESPONSE" | head -n -1
    else
        echo "⚠️  Health check returned: $HTTP_CODE"
    fi
    
else
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "❌ DEPLOYMENT FAILED"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "💡 Troubleshooting tips:"
    echo "1. Check build logs above for errors"
    echo "2. Verify .gcloudignore is excluding large files"
    echo "3. Check GCS permissions: gsutil ls gs://mlops-financial-stress-data/"
    echo "4. View logs: gcloud run logs read financial-stress-api --region us-east1"
    echo ""
    exit 1
fi