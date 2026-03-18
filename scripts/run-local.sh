#!/bin/bash
# Run WeatherScope services locally without Docker
# Usage:
#   ./scripts/run-local.sh              # Show help
#   ./scripts/run-local.sh inference    # Run inference service
#   ./scripts/run-local.sh map          # Run map service
#   ./scripts/run-local.sh llm          # Run LLM service (default: local vllm-metal + Llama 3.2)
#   ./scripts/run-local.sh llm vllm     # Run LLM service with vLLM backend
#   ./scripts/run-local.sh llm model-runner  # Run LLM service via Docker Model Runner
#   ./scripts/run-local.sh fourcastnet  # Run FourCastNet service
#   ./scripts/run-local.sh dashboard    # Run dashboard service
#   ./scripts/run-local.sh --cleanup    # Stop all local services
#   ./scripts/run-local.sh --status     # Show status of local services

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
VLLM_PID_FILE="${VLLM_PID_FILE:-/tmp/weatherscope-vllm.pid}"
VLLM_LOG_FILE="${VLLM_LOG_FILE:-/tmp/weatherscope-vllm.log}"
VLLM_LOCAL_VENV="${VLLM_LOCAL_VENV:-$HOME/.venv-vllm-metal}"
DEFAULT_VLLM_MODEL="${DEFAULT_VLLM_MODEL:-mlx-community/Llama-3.2-3B-Instruct-4bit}"
DEFAULT_MODEL_RUNNER_MODEL="${DEFAULT_MODEL_RUNNER_MODEL:-huggingface.co/mlx-community/llama-3.2-3b-instruct-4bit:latest}"

cd "$PROJECT_ROOT"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

run_inference() {
    echo -e "${GREEN}Starting Inference Service on port 8000...${NC}"
    cd "$PROJECT_ROOT/services/inference"
    $PYTHON -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
}

resolve_vllm_model() {
    local model="${VLLM_SERVER_MODEL:-${LLM_MODEL:-$DEFAULT_VLLM_MODEL}}"
    echo "$model"
}

parse_vllm_host_port() {
    local base="$1"
    base="${base#http://}"
    base="${base#https://}"
    local host_port="${base%%/*}"
    local host="${host_port%%:*}"
    local port="${host_port##*:}"
    if [ "$host" = "$host_port" ]; then
        port=""
    fi
    echo "${host}:${port}"
}

is_vllm_ready() {
    curl -sS --max-time 2 "${VLLM_BASE_URL}${VLLM_API_PREFIX}/models" >/dev/null 2>&1
}

start_local_vllm_if_needed() {
    if is_vllm_ready; then
        echo -e "${GREEN}vLLM server already reachable${NC} at ${VLLM_BASE_URL}${VLLM_API_PREFIX}"
        return 0
    fi

    local host_port
    host_port="$(parse_vllm_host_port "$VLLM_BASE_URL")"
    local host="${host_port%%:*}"
    local port="${host_port##*:}"

    if [ "$host" != "localhost" ] && [ "$host" != "127.0.0.1" ]; then
        echo -e "${YELLOW}Skipping auto-start:${NC} VLLM_BASE_URL is non-local (${VLLM_BASE_URL})"
        return 0
    fi

    if [ -z "$port" ]; then
        echo -e "${RED}Invalid VLLM_BASE_URL (missing port):${NC} ${VLLM_BASE_URL}"
        exit 1
    fi

    if [ ! -x "$VLLM_LOCAL_VENV/bin/vllm" ]; then
        echo -e "${RED}vLLM binary not found at:${NC} $VLLM_LOCAL_VENV/bin/vllm"
        echo "Install first: curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash"
        exit 1
    fi

    local vllm_model
    vllm_model="$(resolve_vllm_model)"
    echo -e "${BLUE}Starting local vLLM server...${NC} model=${vllm_model} port=${port}"
    echo -e "${YELLOW}Log file:${NC} ${VLLM_LOG_FILE}"

    VLLM_PLUGINS="${VLLM_PLUGINS:-metal}" \
    "$VLLM_LOCAL_VENV/bin/vllm" serve "$vllm_model" \
        --host "$host" \
        --port "$port" \
        >"$VLLM_LOG_FILE" 2>&1 &

    local pid=$!
    echo "$pid" >"$VLLM_PID_FILE"

    local i
    for i in $(seq 1 90); do
        if is_vllm_ready; then
            echo -e "${GREEN}vLLM server ready${NC} at ${VLLM_BASE_URL}${VLLM_API_PREFIX}"
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo -e "${RED}vLLM server exited during startup.${NC}"
            echo "Recent log:"
            tail -n 40 "$VLLM_LOG_FILE" 2>/dev/null || true
            rm -f "$VLLM_PID_FILE"
            exit 1
        fi
        sleep 1
    done

    echo -e "${RED}Timed out waiting for vLLM server startup.${NC}"
    echo "Recent log:"
    tail -n 40 "$VLLM_LOG_FILE" 2>/dev/null || true
    exit 1
}

stop_local_vllm() {
    local stopped=false
    if [ -f "$VLLM_PID_FILE" ]; then
        local pid
        pid="$(cat "$VLLM_PID_FILE" 2>/dev/null || true)"
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            sleep 1
            if kill -0 "$pid" 2>/dev/null; then
                kill -9 "$pid" 2>/dev/null || true
            fi
            stopped=true
        fi
        rm -f "$VLLM_PID_FILE"
    fi

    if [ "$stopped" = true ]; then
        echo -e "  ${GREEN}Stopped${NC} vLLM server"
    else
        if pkill -f "vllm serve" 2>/dev/null; then
            echo -e "  ${GREEN}Stopped${NC} vLLM server (matched 'vllm serve')"
        else
            echo "  vLLM server not running"
        fi
    fi
}

run_llm() {
    local backend="${1:-${LLM_BACKEND:-vllm}}"
    backend="$(printf '%s' "$backend" | tr '[:upper:]' '[:lower:]')"

    echo -e "${GREEN}Starting LLM Service on port 8002...${NC}"
    cd "$PROJECT_ROOT/services/llm"

    case "$backend" in
        ollama)
            export LLM_BACKEND="ollama"
            export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
            echo -e "${YELLOW}Backend: ollama (${OLLAMA_BASE_URL})${NC}"
            echo -e "${YELLOW}Note: Run 'ollama serve' in another terminal for full LLM support${NC}"
            ;;
        vllm)
            export LLM_BACKEND="vllm"
            export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8001}"
            export VLLM_API_PREFIX="${VLLM_API_PREFIX:-/v1}"
            export LLM_MODEL="${LLM_MODEL:-$DEFAULT_VLLM_MODEL}"
            export VLLM_SERVER_MODEL="${VLLM_SERVER_MODEL:-$LLM_MODEL}"
            echo -e "${YELLOW}Backend: vllm (${VLLM_BASE_URL}${VLLM_API_PREFIX})${NC}"
            echo -e "${YELLOW}Model:${NC} ${LLM_MODEL}"
            start_local_vllm_if_needed
            ;;
        model-runner|vllm-metal)
            export LLM_BACKEND="vllm"
            export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:12434}"
            export VLLM_API_PREFIX="${VLLM_API_PREFIX:-/engines/v1}"
            export LLM_MODEL="${LLM_MODEL:-$DEFAULT_MODEL_RUNNER_MODEL}"
            echo -e "${YELLOW}Backend: docker-model-runner (${VLLM_BASE_URL}${VLLM_API_PREFIX})${NC}"
            echo -e "${YELLOW}Model:${NC} ${LLM_MODEL}"
            ;;
        *)
            echo -e "${RED}Unknown LLM backend: ${backend}${NC}"
            echo "Use one of: ollama, vllm, model-runner, vllm-metal"
            exit 1
            ;;
    esac

    $PYTHON -m uvicorn app:app --host 0.0.0.0 --port 8002 --reload
}

run_map() {
    echo -e "${GREEN}Starting Map Service on port 8004...${NC}"
    cd "$PROJECT_ROOT/services/map"
    $PYTHON -m uvicorn app:app --host 0.0.0.0 --port 8004 --reload
}

run_fourcastnet() {
    echo -e "${GREEN}Starting FourCastNet Service on port 8003...${NC}"
    echo -e "${YELLOW}Note: Uses MPS acceleration on Apple Silicon${NC}"
    echo -e "${YELLOW}Running FourCastNet asset setup before service start...${NC}"
    bash "$PROJECT_ROOT/scripts/run-fourcastnet.sh"
}

run_dashboard() {
    echo -e "${GREEN}Starting Dashboard Service on port 8010...${NC}"
    cd "$PROJECT_ROOT/services/dashboard"
    FOURCASTNET_URL=http://localhost:8003 \
    INFERENCE_URL=http://localhost:8000 \
    MAP_URL=http://localhost:8004 \
    LLM_URL=http://localhost:8002 \
    $PYTHON -m uvicorn app:app --host 0.0.0.0 --port 8010 --reload
}

cleanup() {
    echo -e "${YELLOW}Stopping local services...${NC}"

    # Kill uvicorn processes for our services
    pkill -f "uvicorn app:app.*8000" 2>/dev/null && echo -e "  ${GREEN}Stopped${NC} inference (8000)" || echo "  Inference (8000) not running"
    pkill -f "uvicorn app:app.*8004" 2>/dev/null && echo -e "  ${GREEN}Stopped${NC} map (8004)" || echo "  Map (8004) not running"
    pkill -f "uvicorn app:app.*8002" 2>/dev/null && echo -e "  ${GREEN}Stopped${NC} LLM (8002)" || echo "  LLM (8002) not running"
    pkill -f "uvicorn app:app.*8003" 2>/dev/null && echo -e "  ${GREEN}Stopped${NC} FourCastNet (8003)" || echo "  FourCastNet (8003) not running"
    pkill -f "uvicorn app:app.*8010" 2>/dev/null && echo -e "  ${GREEN}Stopped${NC} Dashboard (8010)" || echo "  Dashboard (8010) not running"
    stop_local_vllm

    echo -e "${GREEN}Done${NC}"
}

status() {
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}  Local Services Status${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo ""

    # Check each service
    if pgrep -f "uvicorn app:app.*8000" > /dev/null; then
        echo -e "  Inference (8000):   ${GREEN}running${NC}"
    else
        echo -e "  Inference (8000):   ${RED}stopped${NC}"
    fi

    if pgrep -f "uvicorn app:app.*8002" > /dev/null; then
        echo -e "  LLM (8002):         ${GREEN}running${NC}"
    else
        echo -e "  LLM (8002):         ${RED}stopped${NC}"
    fi

    if pgrep -f "uvicorn app:app.*8004" > /dev/null; then
        echo -e "  Map (8004):         ${GREEN}running${NC}"
    else
        echo -e "  Map (8004):         ${RED}stopped${NC}"
    fi

    if pgrep -f "uvicorn app:app.*8003" > /dev/null; then
        echo -e "  FourCastNet (8003): ${GREEN}running${NC}"
    else
        echo -e "  FourCastNet (8003): ${RED}stopped${NC}"
    fi

    if pgrep -f "uvicorn app:app.*8010" > /dev/null; then
        echo -e "  Dashboard (8010):   ${GREEN}running${NC}"
    else
        echo -e "  Dashboard (8010):   ${RED}stopped${NC}"
    fi

    if [ -f "$VLLM_PID_FILE" ] && kill -0 "$(cat "$VLLM_PID_FILE" 2>/dev/null || echo 0)" 2>/dev/null; then
        echo -e "  vLLM (auto):        ${GREEN}running${NC}"
    else
        echo -e "  vLLM (auto):        ${RED}stopped${NC}"
    fi

    echo ""
}

show_help() {
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}  WeatherScope Local Services${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Services:"
    echo "  inference     Start inference service (port 8000)"
    echo "  map           Start map service (port 8004)"
    echo "  llm           Start LLM service (port 8002, default: local vllm-metal + Llama 3.2)"
    echo "  fourcastnet   Start FourCastNet service (port 8003)"
    echo "  dashboard     Start dashboard service (port 8010)"
    echo ""
    echo "Management:"
    echo "  --cleanup, -c Stop all local services"
    echo "  --status, -s  Show status of local services"
    echo "  --help, -h    Show this help"
    echo ""
    echo "Run each service in a separate terminal:"
    echo -e "  ${GREEN}Terminal 1:${NC} ./scripts/run-local.sh inference"
    echo -e "  ${GREEN}Terminal 2:${NC} ./scripts/run-local.sh map"
    echo -e "  ${GREEN}Terminal 3:${NC} ./scripts/run-local.sh llm"
    echo -e "  ${GREEN}Terminal 4:${NC} ./scripts/run-local.sh fourcastnet"
    echo ""
    echo "LLM backend examples:"
    echo "  ./scripts/run-local.sh llm           (default: local vllm-metal)"
    echo "  ./scripts/run-local.sh llm ollama"
    echo "  ./scripts/run-local.sh llm vllm      (auto-starts local vLLM)"
    echo "  ./scripts/run-local.sh llm model-runner"
    echo ""
    echo "Endpoints:"
    echo "  - Inference:   http://localhost:8000  (bilinear/bicubic downscaling)"
    echo "  - Map:         http://localhost:8004/map, /map/regional, /map/downscale"
    echo "  - LLM:         http://localhost:8002  (weather interpretation)"
    echo "  - FourCastNet: http://localhost:8003  (global forecasting)"
    echo "  - Dashboard:   http://localhost:8010  (demo UI)"
    echo ""
    echo "Test with:"
    echo "  curl http://localhost:8000/health"
    echo "  curl http://localhost:8004/health"
    echo "  curl http://localhost:8002/health"
    echo "  curl http://localhost:8003/health"
    echo "  curl http://localhost:8010/health"
}

# Normalize first arg to avoid issues from stray whitespace/CR characters.
ARG_RAW="${1:-help}"
ARG="${ARG_RAW//$'\r'/}"
ARG="${ARG//$'\n'/}"
ARG="${ARG#"${ARG%%[![:space:]]*}"}"
ARG="${ARG%"${ARG##*[![:space:]]}"}"
ARG="$(printf '%s' "$ARG" | tr '[:upper:]' '[:lower:]')"

case "$ARG" in
    inference)
        run_inference
        ;;
    map)
        run_map
        ;;
    llm)
        run_llm "${2:-}"
        ;;
    fourcastnet)
        run_fourcastnet
        ;;
    dashboard)
        run_dashboard
        ;;
    --cleanup|-c)
        cleanup
        ;;
    --status|-s)
        status
        ;;
    --help|-h|help)
        show_help
        ;;
    *)
        echo "Unknown option: ${ARG_RAW}"
        echo "Use --help for usage information"
        exit 1
        ;;
esac
