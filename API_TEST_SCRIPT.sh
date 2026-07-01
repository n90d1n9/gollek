#!/bin/bash

# Gollek API Testing Script
# Tests all major endpoints with proper error handling

BASE_URL="${1:-http://localhost:9131}"
API_KEY="${2:-community}"
VERBOSE="${3:-false}"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

test_count=0
pass_count=0
fail_count=0

# Helper function for testing
test_endpoint() {
    local method=$1
    local endpoint=$2
    local data=$3
    local expected_status=$4
    local description=$5
    
    test_count=$((test_count + 1))
    
    # Use temp file to separate HTTP code from response body
    tmpfile=$(mktemp)
    trap "rm -f $tmpfile" EXIT
    
    if [ -n "$data" ]; then
        curl -s -w "%{http_code}" -X "$method" \
            -H 'Accept: application/json' \
            -H 'Content-Type: application/json' \
            -H "X-API-Key: $API_KEY" \
            -d "$data" \
            "$BASE_URL$endpoint" > "$tmpfile"
    else
        curl -s -w "%{http_code}" -X "$method" \
            -H 'Accept: application/json' \
            -H "X-API-Key: $API_KEY" \
            "$BASE_URL$endpoint" > "$tmpfile"
    fi
    
    # Extract HTTP code from last 3 chars
    http_code=$(tail -c 3 "$tmpfile")
    body=$(sed '$s/...//' "$tmpfile")
    
    if [ "$http_code" = "$expected_status" ]; then
        echo -e "${GREEN}✓${NC} Test $test_count: $description (HTTP $http_code)"
        pass_count=$((pass_count + 1))
        
        if [ "$VERBOSE" = "true" ] && [ -n "$body" ]; then
            echo "  Response: $(echo "$body" | head -c 100)..."
        fi
    else
        echo -e "${RED}✗${NC} Test $test_count: $description (Expected $expected_status, got $http_code)"
        fail_count=$((fail_count + 1))
        
        if [ "$VERBOSE" = "true" ]; then
            echo "  Response: $body"
        fi
    fi
}

echo "======================================"
echo "Gollek API Test Suite"
echo "======================================"
echo "Base URL: $BASE_URL"
echo "API Key: ${API_KEY:0:8}..."
echo ""

# Health Check
echo "🏥 Health Check"
test_endpoint "GET" "/health" "" "200" "Server health check"
echo ""

# Models Endpoints
echo "📦 Models Management"
test_endpoint "GET" "/v1/models" "" "200" "List all models"
test_endpoint "GET" "/v1/models?limit=5&runnableOnly=true" "" "200" "List runnable models (limited)"
test_endpoint "GET" "/v1/models?compat=openai" "" "200" "List models in OpenAI format"
echo ""

# Chat Completions
echo "💬 Chat Completions"
test_endpoint "POST" "/v1/chat/completions" \
    '{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}' \
    "200" "Simple chat completion"
echo ""

# Embeddings
echo "🔌 Embeddings"
test_endpoint "POST" "/v1/embeddings" \
    '{"model":"text-embedding-ada-002","input":"test text"}' \
    "200" "Create embedding"
echo ""

# System Info
echo "🔍 System Info"
test_endpoint "GET" "/v1/system" "" "200" "Get system information"
echo ""

# Providers
echo "📋 Providers"
test_endpoint "GET" "/v1/providers" "" "200" "List available providers"
echo ""

# Authentication Tests
echo "🔐 Authentication"
test_endpoint "GET" "/v1/models" "" "401" "Missing API key (should return 401)"
echo ""

# Results
echo "======================================"
echo "Test Results: ${GREEN}$pass_count passed${NC}, ${RED}$fail_count failed${NC} out of $test_count tests"
echo "======================================"

if [ $fail_count -eq 0 ]; then
    exit 0
else
    exit 1
fi
