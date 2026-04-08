# Production Deployment Script

#!/bin/bash

set -e

# Configuration
NAMESPACE="multimodal-production"
IMAGE_NAME="multimodal-inference"
IMAGE_TAG="v1.0.0"
REGISTRY="registry.example.com"

echo "========================================="
echo "Multimodal Inference Production Deployment"
echo "========================================="
echo ""

# Step 1: Pre-deployment checks
echo "Step 1: Pre-deployment checks..."
echo "-----------------------------------------"

# Check kubectl access
if ! kubectl cluster-info &> /dev/null; then
    echo "ERROR: Cannot connect to Kubernetes cluster"
    exit 1
fi
echo "✓ Kubernetes cluster accessible"

# Check namespace exists
if ! kubectl get namespace $NAMESPACE &> /dev/null; then
    echo "Creating namespace: $NAMESPACE"
    kubectl create namespace $NAMESPACE
fi
echo "✓ Namespace ready"

# Check image exists
if ! docker pull ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} &> /dev/null; then
    echo "WARNING: Image not found in registry. Building..."
    # Build and push image
    cd ../../../../../../
    mvn clean package -DskipTests
    docker build -t ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} .
    docker push ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}
fi
echo "✓ Image available: ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"

echo ""
echo "Step 2: Apply Kubernetes manifests..."
echo "-----------------------------------------"

# Apply namespace and config
kubectl apply -f src/main/resources/kubernetes/production/deployment.yaml

# Wait for deployment to start
echo "Waiting for deployment to start..."
kubectl rollout status deployment/multimodal-inference -n $NAMESPACE --timeout=300s

echo ""
echo "Step 3: Verify deployment..."
echo "-----------------------------------------"

# Check pods are running
echo "Checking pod status..."
kubectl get pods -n $NAMESPACE -l app=multimodal-inference

# Check service is available
echo "Checking service..."
kubectl get svc -n $NAMESPACE multimodal-inference

# Check health endpoint
echo "Checking health endpoint..."
sleep 10  # Wait for pods to be ready
POD_NAME=$(kubectl get pods -n $NAMESPACE -l app=multimodal-inference -o jsonpath="{.items[0].metadata.name}")
kubectl exec -n $NAMESPACE $POD_NAME -- curl -s http://localhost:8080/q/health | jq .

echo ""
echo "Step 4: Run smoke tests..."
echo "-----------------------------------------"

# Run basic smoke test
cat > /tmp/smoke-test.json << 'EOF'
{
  "model": "clip-vit-base",
  "inputs": [
    {
      "modality": "TEXT",
      "text": "test"
    }
  ]
}
EOF

echo "Running smoke test..."
kubectl exec -n $NAMESPACE $POD_NAME -- curl -s -X POST http://localhost:8080/api/infer \
  -H "Content-Type: application/json" \
  -d @/tmp/smoke-test.json | jq .

echo ""
echo "Step 5: Update monitoring..."
echo "-----------------------------------------"

# Annotate deployment for monitoring
kubectl annotate deployment multimodal-inference -n $NAMESPACE \
  prometheus.io/scrape="true" \
  prometheus.io/port="8080" \
  prometheus.io/path="/metrics" \
  --overwrite

echo "✓ Monitoring configured"

echo ""
echo "Step 6: Post-deployment validation..."
echo "-----------------------------------------"

# Get deployment status
echo "Deployment status:"
kubectl get deployment multimodal-inference -n $NAMESPACE -o wide

# Get HPA status
echo "HPA status:"
kubectl get hpa -n $NAMESPACE

# Get pod metrics
echo "Pod metrics:"
kubectl top pods -n $NAMESPACE -l app=multimodal-inference

echo ""
echo "========================================="
echo "Deployment Complete!"
echo "========================================="
echo ""
echo "Service URL: $(kubectl get svc multimodal-inference -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')"
echo "Health Check: http://$(kubectl get svc multimodal-inference -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')/q/health"
echo "Metrics: http://$(kubectl get svc multimodal-inference -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')/metrics"
echo ""
echo "Next steps:"
echo "1. Monitor pod health: kubectl get pods -n $NAMESPACE -w"
echo "2. Check logs: kubectl logs -n $NAMESPACE -l app=multimodal-inference -f"
echo "3. View metrics: kubectl port-forward svc/multimodal-inference 9090:80 -n $NAMESPACE"
echo ""
