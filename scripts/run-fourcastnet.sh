#!/bin/bash
# Run FourCastNet v2-small service natively on macOS with MPS
# This allows GPU acceleration while integrating with k8s Gateway

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
if [ -z "${PYTHON:-}" ]; then
    if [ -x "/opt/miniconda3/envs/weatherscope/bin/python" ]; then
        PYTHON="/opt/miniconda3/envs/weatherscope/bin/python"
    elif [ -x "/opt/miniconda3/envs/regionalcast/bin/python" ]; then
        PYTHON="/opt/miniconda3/envs/regionalcast/bin/python"
    else
        PYTHON="python3"
    fi
fi

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Check dependencies
check_deps() {
    echo -e "\n${YELLOW}Checking dependencies...${NC}"

    # Check ai-models
    if ! $PYTHON -c "import ai_models" 2>/dev/null; then
        echo -e "${YELLOW}Installing ai-models...${NC}"
        $PYTHON -m pip install ai-models ai-models-fourcastnetv2
    fi
    echo -e "${GREEN}✓ ai-models installed${NC}"

    # Check FastAPI
    if ! $PYTHON -c "import fastapi" 2>/dev/null; then
        echo -e "${YELLOW}Installing FastAPI...${NC}"
        $PYTHON -m pip install fastapi uvicorn
    fi
    echo -e "${GREEN}✓ FastAPI installed${NC}"

    # Check for cfgrib (optional, for reading output)
    if ! $PYTHON -c "import cfgrib" 2>/dev/null; then
        echo -e "${YELLOW}Note: cfgrib not installed. Install for GRIB reading:${NC}"
        echo "  $PYTHON -m pip install cfgrib eccodes"
    fi
}

# Setup model assets
setup_assets() {
    echo -e "\n${YELLOW}Checking model assets...${NC}"

    CACHE_DIR="$HOME/.cache/ai-models/fourcastnetv2-small"
    WEIGHTS_FILE="$CACHE_DIR/weights.tar"
    GLOBAL_MEANS_FILE="$CACHE_DIR/global_means.npy"
    GLOBAL_STDS_FILE="$CACHE_DIR/global_stds.npy"
    LOCAL_WEIGHTS="$PROJECT_ROOT/fourcastnet/weights.tar"
    LOCAL_GLOBAL_MEANS="$PROJECT_ROOT/fourcastnet/global_means.npy"
    LOCAL_GLOBAL_STDS="$PROJECT_ROOT/fourcastnet/global_stds.npy"

    # Check if required assets already exist in cache
    if [ -f "$WEIGHTS_FILE" ] && [ -f "$GLOBAL_MEANS_FILE" ] && [ -f "$GLOBAL_STDS_FILE" ]; then
        echo -e "${GREEN}✓ Model assets found in cache:${NC}"
        echo "  - $WEIGHTS_FILE"
        echo "  - $GLOBAL_MEANS_FILE"
        echo "  - $GLOBAL_STDS_FILE"
        return 0
    fi

    # Try local asset copies in project/fourcastnet
    if [ -f "$LOCAL_WEIGHTS" ] || [ -f "$LOCAL_GLOBAL_MEANS" ] || [ -f "$LOCAL_GLOBAL_STDS" ]; then
        echo -e "${YELLOW}Found local model assets in: $PROJECT_ROOT/fourcastnet${NC}"
        echo -e "${YELLOW}Copying to cache directory...${NC}"

        # Create cache directory
        mkdir -p "$CACHE_DIR"

        # Copy weights if missing in cache
        if [ ! -f "$WEIGHTS_FILE" ] && [ -f "$LOCAL_WEIGHTS" ]; then
            cp "$LOCAL_WEIGHTS" "$WEIGHTS_FILE"
            if [ -f "$WEIGHTS_FILE" ]; then
                echo -e "${GREEN}✓ Copied weights: $WEIGHTS_FILE${NC}"
            else
                echo -e "${YELLOW}Failed to copy weights${NC}"
            fi
        fi

        # Copy global means if missing in cache
        if [ ! -f "$GLOBAL_MEANS_FILE" ] && [ -f "$LOCAL_GLOBAL_MEANS" ]; then
            cp "$LOCAL_GLOBAL_MEANS" "$GLOBAL_MEANS_FILE"
            if [ -f "$GLOBAL_MEANS_FILE" ]; then
                echo -e "${GREEN}✓ Copied global means: $GLOBAL_MEANS_FILE${NC}"
            else
                echo -e "${YELLOW}Failed to copy global_means.npy${NC}"
            fi
        fi

        # Copy global stds if missing in cache
        if [ ! -f "$GLOBAL_STDS_FILE" ] && [ -f "$LOCAL_GLOBAL_STDS" ]; then
            cp "$LOCAL_GLOBAL_STDS" "$GLOBAL_STDS_FILE"
            if [ -f "$GLOBAL_STDS_FILE" ]; then
                echo -e "${GREEN}✓ Copied global stds: $GLOBAL_STDS_FILE${NC}"
            else
                echo -e "${YELLOW}Failed to copy global_stds.npy${NC}"
            fi
        fi

        # Re-check cache after copy attempts
        if [ -f "$WEIGHTS_FILE" ] && [ -f "$GLOBAL_MEANS_FILE" ] && [ -f "$GLOBAL_STDS_FILE" ]; then
            return 0
        fi
    fi

    # Missing one or more required files - print instructions
    echo -e "${YELLOW}Model assets are incomplete.${NC}"
    [ ! -f "$WEIGHTS_FILE" ] && echo "  Missing: $WEIGHTS_FILE"
    [ ! -f "$GLOBAL_MEANS_FILE" ] && echo "  Missing: $GLOBAL_MEANS_FILE"
    [ ! -f "$GLOBAL_STDS_FILE" ] && echo "  Missing: $GLOBAL_STDS_FILE"
    echo ""
    echo "Options:"
    echo "  1. Place all files in: $PROJECT_ROOT/fourcastnet/"
    echo "     - weights.tar"
    echo "     - global_means.npy"
    echo "     - global_stds.npy"
    echo "  2. Download via: ai-models --download-assets fourcastnetv2-small"
    echo "  3. Assets will auto-download on first forecast (requires network)"
    echo ""
    echo -e "${BLUE}Data source: CDS (Climate Data Store)${NC}"
    echo -e "${YELLOW}Requirements for auto-download:${NC}"
    echo "  1. CDS account: https://cds.climate.copernicus.eu/"
    echo "  2. Accept ERA5 license on CDS website"
    echo "  3. API key in ~/.cdsapirc:"
    echo "     url: https://cds.climate.copernicus.eu/api"
    echo "     key: <your-uid>:<your-api-key>"
    echo ""
    echo -e "${YELLOW}Note: ERA5 data has ~5 day lag from current date${NC}"
}

# Alias for backward compatibility
download_assets() {
    setup_assets
}

# Run service
run_service() {
    echo -e "\n${GREEN}Starting FourCastNet service on port 8003...${NC}"
    echo -e "${YELLOW}MPS (Metal) will be used if available${NC}"
    echo ""
    echo "Endpoints:"
    echo "  - Health:    http://localhost:8003/health"
    echo "  - Forecast:  POST http://localhost:8003/forecast"
    echo "  - Status:    GET http://localhost:8003/forecast/{job_id}"
    echo "  - Map:       GET http://localhost:8003/forecast/{job_id}/map?variable=t2m&step=0"
    echo "  - Regional:  GET http://localhost:8003/forecast/{job_id}/regional?region=NC&step=0"
    echo "               POST http://localhost:8004/map/regional (body: regional JSON)"
    echo ""
    echo -e "${YELLOW}Gateway integration:${NC}"
    echo "  kubectl apply -f k8s/fourcastnet-service.yaml"
    echo "  kubectl apply -f k8s/gateway.yaml"
    echo "  Then access via:"
    echo "    - http://localhost:8088/forecast  (default run-k8s port-forward)"
    echo "    - http://localhost:8080/forecast  (k3d load balancer)"
    echo ""

    cd "$PROJECT_ROOT/services/fourcastnet"
    $PYTHON -m uvicorn app:app --host 0.0.0.0 --port 8003 --reload
}

# Quick test (using CDS as source)
test_forecast() {
    echo -e "\n${YELLOW}Running quick 1-day forecast test...${NC}"
    echo -e "${YELLOW}Using CDS (Climate Data Store) as data source${NC}"
    echo -e "${YELLOW}Note: ERA5 data has ~5 day lag, using date from 5 days ago${NC}"

    # Calculate date 5 days ago for ERA5 availability
    FORECAST_DATE=$(date -v-5d +%Y%m%d 2>/dev/null || date -d "5 days ago" +%Y%m%d)

    echo -e "${BLUE}Forecast date: $FORECAST_DATE${NC}"

    # Use CDS input and set environment variable for PyTorch weights
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 ai-models \
        --input cds \
        --date "$FORECAST_DATE" \
        --time 1200 \
        --lead-time 24 \
        fourcastnetv2-small

    echo -e "${GREEN}✓ Test complete${NC}"
}

# Cleanup - stop FourCastNet service
cleanup() {
    echo -e "${YELLOW}Stopping FourCastNet service...${NC}"

    if pkill -f "uvicorn app:app.*8003" 2>/dev/null; then
        echo -e "  ${GREEN}Stopped${NC} FourCastNet (8003)"
    else
        echo "  FourCastNet (8003) not running"
    fi

    echo -e "${GREEN}Done${NC}"
}

# Status - check if FourCastNet is running
status() {
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}  FourCastNet Status${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo ""

    if pgrep -f "uvicorn app:app.*8003" > /dev/null; then
        echo -e "  FourCastNet (8003): ${GREEN}running${NC}"
        echo ""
        echo "  Test: curl http://localhost:8003/health"
    else
        echo -e "  FourCastNet (8003): ${RED}stopped${NC}"
        echo ""
        echo "  Start with: ./scripts/run-fourcastnet.sh"
    fi

    echo ""
}

# Print header
print_header() {
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}  FourCastNet v2-small Service (MPS)${NC}"
    echo -e "${BLUE}================================================${NC}"
}

# Main
case "${1:-}" in
    --cleanup|-c)
        cleanup
        exit 0
        ;;
    --status)
        status
        exit 0
        ;;
    --deps|-d)
        print_header
        check_deps
        ;;
    --download|-a)
        print_header
        check_deps
        download_assets
        ;;
    --test|-t)
        print_header
        check_deps
        download_assets
        test_forecast
        ;;
    --service|-s)
        # Start service directly without asset check
        print_header
        check_deps
        run_service
        ;;
    --help|-h)
        print_header
        echo ""
        echo "Usage: $0 [OPTION]"
        echo ""
        echo "Start service:"
        echo "  (none)          Check deps, assets, then start service"
        echo "  --service, -s   Start service directly (skip asset check)"
        echo ""
        echo "Management:"
        echo "  --cleanup, -c   Stop FourCastNet service"
        echo "  --status        Show service status"
        echo ""
        echo "Setup:"
        echo "  --deps, -d      Check dependencies only"
        echo "  --download, -a  Check/download model assets"
        echo "  --test, -t      Run a quick test forecast"
        echo "  --help, -h      Show this help"
        ;;
    *)
        print_header
        check_deps
        download_assets
        run_service
        ;;
esac
