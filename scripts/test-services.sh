#!/bin/bash
# Test script for WeatherScope services
# Run after docker-compose up or k8s deployment

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Help function
show_help() {
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}  WeatherScope Services Test Suite${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo ""
    echo "Usage: $0 [OPTIONS] [INFERENCE_URL] [LLM_URL] [FOURCASTNET_URL]"
    echo ""
    echo "Options:"
    echo "  --help, -h      Show this help"
    echo "  --gateway, -g   Test via gateway (http://localhost:8080)"
    echo "  --local, -l     Test local services (default)"
    echo ""
    echo "Arguments (optional):"
    echo "  INFERENCE_URL   Inference service URL (default: http://localhost:8000)"
    echo "  LLM_URL         LLM service URL (default: http://localhost:8002)"
    echo "  FOURCASTNET_URL FourCastNet service URL (default: http://localhost:8003)"
    echo ""
    echo "Examples:"
    echo "  $0                           # Test local services"
    echo "  $0 --gateway                 # Test via K8s gateway"
    echo "  $0 http://custom:8000        # Custom inference URL"
}

# Parse arguments
case "${1:-}" in
    --help|-h)
        show_help
        exit 0
        ;;
    --gateway|-g)
        GATEWAY_URL="http://localhost:8080"
        INFERENCE_URL="$GATEWAY_URL"
        LLM_URL="$GATEWAY_URL"
        FOURCASTNET_URL="$GATEWAY_URL"
        echo -e "${YELLOW}Testing via gateway: $GATEWAY_URL${NC}"
        ;;
    --local|-l)
        INFERENCE_URL="${2:-http://localhost:8000}"
        LLM_URL="${3:-http://localhost:8002}"
        FOURCASTNET_URL="${4:-http://localhost:8003}"
        ;;
    *)
        INFERENCE_URL="${1:-http://localhost:8000}"
        LLM_URL="${2:-http://localhost:8002}"
        FOURCASTNET_URL="${3:-http://localhost:8003}"
        ;;
esac

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  WeatherScope Services Test Suite${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo "Testing endpoints:"
echo "  Inference:   $INFERENCE_URL"
echo "  LLM:         $LLM_URL"
echo "  FourCastNet: $FOURCASTNET_URL"
echo ""

# Test function
test_endpoint() {
    local name="$1"
    local url="$2"
    local method="${3:-GET}"
    local data="$4"

    echo -n "Testing $name... "

    if [ "$method" == "POST" ]; then
        response=$(curl -s -w "\n%{http_code}" -X POST -H "Content-Type: application/json" -d "$data" "$url" 2>/dev/null)
    else
        response=$(curl -s -w "\n%{http_code}" "$url" 2>/dev/null)
    fi

    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')

    if [ "$http_code" -ge 200 ] && [ "$http_code" -lt 300 ]; then
        echo -e "${GREEN}✓ PASS${NC} (HTTP $http_code)"
        return 0
    else
        echo -e "${RED}✗ FAIL${NC} (HTTP $http_code)"
        echo "  Response: $body"
        return 1
    fi
}

echo "----------------------------------------"
echo "1. Inference Service Tests"
echo "----------------------------------------"

test_endpoint "Health check" "$INFERENCE_URL/health"
test_endpoint "Root endpoint" "$INFERENCE_URL/"

# Test downscaling with sample data
echo -n "Testing downscale (bilinear)... "
DOWNSCALE_DATA='{"variables": {"t2m": [[280,281,282],[281,282,283],[282,283,284]], "u10": [[1,2,3],[2,3,4],[3,4,5]]}, "upscale_factor": 2, "method": "bilinear"}'
response=$(curl -s -X POST -H "Content-Type: application/json" -d "$DOWNSCALE_DATA" "$INFERENCE_URL/downscale" 2>/dev/null)
if echo "$response" | grep -q "predictions"; then
    echo -e "${GREEN}✓ PASS${NC}"
else
    echo -e "${RED}✗ FAIL${NC}"
    echo "  Response: $response"
fi

echo ""
echo "----------------------------------------"
echo "2. LLM Service Tests"
echo "----------------------------------------"

test_endpoint "Health check" "$LLM_URL/health"
test_endpoint "Config" "$LLM_URL/config"

# Test interpretation
echo -n "Testing interpretation... "
INTERPRET_DATA='{
    "region": "austin",
    "forecast_hour": 6,
    "max_precipitation": 45.5,
    "severity": "severe",
    "affected_percentage": 15.3
}'
response=$(curl -s -X POST -H "Content-Type: application/json" -d "$INTERPRET_DATA" "$LLM_URL/interpret" 2>/dev/null)
if echo "$response" | grep -q "alert"; then
    echo -e "${GREEN}✓ PASS${NC}"
else
    echo -e "${YELLOW}⚠ WARNING${NC} (LLM may not be available)"
fi

echo ""
echo "----------------------------------------"
echo "3. FourCastNet Service Tests"
echo "----------------------------------------"

test_endpoint "Health check" "$FOURCASTNET_URL/health"
test_endpoint "List regions" "$FOURCASTNET_URL/regions"
test_endpoint "Cache status" "$FOURCASTNET_URL/cache"

echo ""
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Test Summary${NC}"
echo -e "${BLUE}================================================${NC}"
echo -e "${GREEN}Tests completed!${NC}"
echo ""
echo "To run with Ollama for full LLM support:"
echo "  1. Install Ollama: brew install ollama"
echo "  2. Start Ollama: ollama serve"
echo "  3. Pull model: ollama pull llama3.2:3b"
echo ""
echo "To start a forecast:"
echo '  curl -X POST http://localhost:8003/forecast \'
echo '    -H "Content-Type: application/json" \'
echo '    -d '\''{"lead_time": 24}'\'''
