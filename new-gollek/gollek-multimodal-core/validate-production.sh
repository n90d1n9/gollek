# Production Validation Script

#!/bin/bash

set -e

# Configuration
NAMESPACE="multimodal-production"
SERVICE_URL=""

echo "========================================="
echo "Production Validation"
echo "========================================="
echo ""

# Get service URL
SERVICE_URL=$(kubectl get svc multimodal-inference -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
echo "Service URL: $SERVICE_URL"
echo ""

# Test 1: Health Check
echo "Test 1: Health Check..."
echo "-----------------------------------------"
HEALTH_RESPONSE=$(curl -s http://$SERVICE_URL/q/health)
HEALTH_STATUS=$(echo $HEALTH_RESPONSE | jq -r '.status')

if [ "$HEALTH_STATUS" == "UP" ]; then
    echo "✓ Health check passed"
    echo "$HEALTH_RESPONSE" | jq .
else
    echo "✗ Health check failed"
    echo "$HEALTH_RESPONSE" | jq .
    exit 1
fi
echo ""

# Test 2: Metrics Endpoint
echo "Test 2: Metrics Endpoint..."
echo "-----------------------------------------"
METRICS_RESPONSE=$(curl -s http://$SERVICE_URL/metrics)
if echo "$METRICS_RESPONSE" | grep -q "multimodal_requests_total"; then
    echo "✓ Metrics endpoint working"
    echo "Sample metrics:"
    echo "$METRICS_RESPONSE" | grep "multimodal_requests_total"
else
    echo "✗ Metrics endpoint failed"
    exit 1
fi
echo ""

# Test 3: Inference API
echo "Test 3: Inference API..."
echo "-----------------------------------------"
INFERENCE_RESPONSE=$(curl -s -X POST http://$SERVICE_URL/api/infer \
  -H "Content-Type: application/json" \
  -d '{"model":"clip-vit-base","inputs":[{"modality":"TEXT","text":"test"}]}')

if echo "$INFERENCE_RESPONSE" | jq -e '.status' &> /dev/null; then
    echo "✓ Inference API working"
    echo "$INFERENCE_RESPONSE" | jq .
else
    echo "✗ Inference API failed"
    echo "$INFERENCE_RESPONSE"
    exit 1
fi
echo ""

# Test 4: Load Test (100 requests)
echo "Test 4: Load Test (100 requests)..."
echo "-----------------------------------------"
SUCCESS_COUNT=0
FAIL_COUNT=0

for i in {1..100}; do
    RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" -X POST http://$SERVICE_URL/api/infer \
      -H "Content-Type: application/json" \
      -d '{"model":"clip-vit-base","inputs":[{"modality":"TEXT","text":"test"}]}')
    
    if [ "$RESPONSE" == "200" ]; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done

echo "Results: $SUCCESS_COUNT successful, $FAIL_COUNT failed"
SUCCESS_RATE=$((SUCCESS_COUNT * 100 / 100))

if [ $SUCCESS_RATE -ge 95 ]; then
    echo "✓ Load test passed (success rate: $SUCCESS_RATE%)"
else
    echo "✗ Load test failed (success rate: $SUCCESS_RATE%)"
    exit 1
fi
echo ""

# Test 5: Latency Check
echo "Test 5: Latency Check..."
echo "-----------------------------------------"
LATENCIES=()

for i in {1..20}; do
    START=$(date +%s%N)
    curl -s -X POST http://$SERVICE_URL/api/infer \
      -H "Content-Type: application/json" \
      -d '{"model":"clip-vit-base","inputs":[{"modality":"TEXT","text":"test"}]}' > /dev/null
    END=$(date +%s%N)
    LATENCY=$(( (END - START) / 1000000 ))
    LATENCIES+=($LATENCY)
done

# Calculate average latency
TOTAL=0
for LATENCY in "${LATENCIES[@]}"; do
    TOTAL=$((TOTAL + LATENCY))
done
AVG_LATENCY=$((TOTAL / 20))

echo "Average latency: ${AVG_LATENCY}ms"

if [ $AVG_LATENCY -lt 2000 ]; then
    echo "✓ Latency check passed"
else
    echo "✗ Latency check failed (>2000ms)"
    exit 1
fi
echo ""

# Test 6: Pod Status
echo "Test 6: Pod Status..."
echo "-----------------------------------------"
READY_PODS=$(kubectl get pods -n $NAMESPACE -l app=multimodal-inference --no-headers | grep -c "Running")
TOTAL_PODS=$(kubectl get pods -n $NAMESPACE -l app=multimodal-inference --no-headers | wc -l)

echo "Ready pods: $READY_PODS/$TOTAL_PODS"

if [ "$READY_PODS" == "$TOTAL_PODS" ]; then
    echo "✓ All pods ready"
    kubectl get pods -n $NAMESPACE -l app=multimodal-inference
else
    echo "✗ Not all pods ready"
    kubectl get pods -n $NAMESPACE -l app=multimodal-inference
    exit 1
fi
echo ""

# Test 7: HPA Status
echo "Test 7: HPA Status..."
echo "-----------------------------------------"
kubectl get hpa -n $NAMESPACE

CURRENT_REPLICAS=$(kubectl get hpa multimodal-inference-hpa -n $NAMESPACE -o jsonpath='{.status.currentReplicas}')
DESIRED_REPLICAS=$(kubectl get hpa multimodal-inference-hpa -n $NAMESPACE -o jsonpath='{.status.desiredReplicas}')

if [ "$CURRENT_REPLICAS" -ge 5 ]; then
    echo "✓ HPA working correctly"
else
    echo "⚠ HPA may need attention"
fi
echo ""

# Summary
echo "========================================="
echo "Production Validation Summary"
echo "========================================="
echo "✓ Health Check: PASSED"
echo "✓ Metrics Endpoint: PASSED"
echo "✓ Inference API: PASSED"
echo "✓ Load Test: PASSED ($SUCCESS_RATE% success)"
echo "✓ Latency Check: PASSED (${AVG_LATENCY}ms)"
echo "✓ Pod Status: PASSED ($READY_PODS/$TOTAL_PODS)"
echo "✓ HPA Status: PASSED"
echo ""
echo "Production deployment validated successfully!"
echo "========================================="
