#!/bin/bash
# WeatherScope Kubernetes Deployment
# Unified script that intelligently manages cluster, gateway, and services
#
# Usage:
#   ./scripts/run-k8s.sh              # Full setup (creates what's missing)
#   ./scripts/run-k8s.sh --status     # Show current status
#   ./scripts/run-k8s.sh --rebuild    # Rebuild and redeploy images
#   ./scripts/run-k8s.sh --gateway    # Redeploy gateway config only
#   ./scripts/run-k8s.sh --cleanup    # Delete cluster

set -e

# Configuration
CLUSTER_NAME="weatherscope"
KGATEWAY_VERSION="v2.2.0-rc.2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
K8S_DIR="$PROJECT_ROOT/k8s"
SERVICES_DIR="$PROJECT_ROOT/services"
PORT_FORWARD_ENABLED=true
PORT_FORWARD_PORT="8088"
K3D_LB_PORT="${K3D_LB_PORT:-8080}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging helpers
info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

is_port_in_use() {
    local port="$1"
    lsof -nP -iTCP:"${port}" -sTCP:LISTEN &> /dev/null
}

# ============================================================================
# Prerequisites Check
# ============================================================================
check_prerequisites() {
    info "Checking prerequisites..."
    local missing=0

    # Docker
    if command -v docker &> /dev/null && docker info &> /dev/null; then
        success "Docker running"
    else
        error "Docker not running or not installed"
        echo "  Install from: https://www.docker.com/products/docker-desktop"
        missing=1
    fi

    # k3d
    if command -v k3d &> /dev/null; then
        success "k3d installed ($(k3d version | head -1 | awk '{print $3}'))"
    else
        warn "k3d not found - will install"
        if [[ "$OSTYPE" == "darwin"* ]]; then
            brew install k3d
        else
            curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | bash
        fi
        success "k3d installed"
    fi

    # kubectl
    if command -v kubectl &> /dev/null; then
        success "kubectl installed"
    else
        warn "kubectl not found - will install"
        if [[ "$OSTYPE" == "darwin"* ]]; then
            brew install kubectl
        else
            curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
            chmod +x kubectl && sudo mv kubectl /usr/local/bin/
        fi
        success "kubectl installed"
    fi

    # Helm
    if command -v helm &> /dev/null; then
        success "helm installed"
    else
        warn "helm not found - will install"
        if [[ "$OSTYPE" == "darwin"* ]]; then
            brew install helm
        else
            curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
        fi
        success "helm installed"
    fi

    if [ $missing -eq 1 ]; then
        error "Missing prerequisites. Please fix and retry."
        exit 1
    fi
}

# ============================================================================
# Cluster Management
# ============================================================================
ensure_cluster() {
    info "Checking k3d cluster..."

    if k3d cluster list 2>/dev/null | grep -q "$CLUSTER_NAME"; then
        success "Cluster '$CLUSTER_NAME' exists"

        # Ensure kubectl context is set
        kubectl config use-context "k3d-$CLUSTER_NAME" &> /dev/null || true

        if kubectl cluster-info &> /dev/null; then
            success "Cluster is accessible"
        else
            warn "Cluster exists but not accessible, restarting..."
            k3d cluster start "$CLUSTER_NAME"
            success "Cluster restarted"
        fi
    else
        info "Creating cluster '$CLUSTER_NAME'..."
        if is_port_in_use "$K3D_LB_PORT"; then
            error "Cannot create cluster: host port ${K3D_LB_PORT} is already in use"
            lsof -nP -iTCP:"${K3D_LB_PORT}" -sTCP:LISTEN || true
            echo "  Retry with: $0 --lb-port <free-port>"
            exit 1
        fi
        k3d cluster create "$CLUSTER_NAME" \
            --port "${K3D_LB_PORT}:80@loadbalancer" \
            --agents 1 \
            --wait
        kubectl config use-context "k3d-$CLUSTER_NAME" &> /dev/null || true
        success "Cluster created"
    fi
}

# ============================================================================
# Kgateway Setup
# ============================================================================
ensure_kgateway() {
    info "Checking Kgateway installation..."

    # Check if Gateway API CRDs exist
    if kubectl get crd gateways.gateway.networking.k8s.io &> /dev/null; then
        success "Gateway API CRDs installed"
    else
        info "Installing Gateway API CRDs..."
        kubectl apply -f https://github.com/kubernetes-sigs/gateway-api/releases/download/v1.2.0/standard-install.yaml
        success "Gateway API CRDs installed"
    fi

    # Check if Kgateway CRDs exist
    if kubectl get crd inferencepools.inference.networking.x-k8s.io &> /dev/null 2>&1; then
        success "Kgateway CRDs installed"
    else
        info "Installing Kgateway CRDs..."
        helm install kgateway-crds oci://cr.kgateway.dev/kgateway-dev/charts/kgateway-crds \
            --version "$KGATEWAY_VERSION" \
            -n kgateway-system --create-namespace \
            --wait 2>/dev/null || warn "Kgateway CRDs may already exist"
        success "Kgateway CRDs installed"
    fi

    # Check if Kgateway controller is running
    if kubectl get deployment kgateway -n kgateway-system &> /dev/null; then
        if kubectl get deployment kgateway -n kgateway-system -o jsonpath='{.status.availableReplicas}' 2>/dev/null | grep -q "1"; then
            success "Kgateway controller running"
        else
            warn "Kgateway controller not ready, waiting..."
            kubectl wait --timeout=2m -n kgateway-system deployment/kgateway \
                --for=condition=Available 2>/dev/null || warn "Still starting..."
        fi
    else
        info "Installing Kgateway controller..."
        helm install kgateway oci://cr.kgateway.dev/kgateway-dev/charts/kgateway \
            --version "$KGATEWAY_VERSION" \
            -n kgateway-system \
            --wait 2>/dev/null || warn "Kgateway may already exist"

        info "Waiting for Kgateway controller..."
        kubectl wait --timeout=2m -n kgateway-system deployment/kgateway \
            --for=condition=Available 2>/dev/null || warn "Controller still starting"
        success "Kgateway controller installed"
    fi
}

# ============================================================================
# Docker Images
# ============================================================================
build_images() {
    local force="${1:-false}"
    info "Checking Docker images..."

    # Check if images exist
    local need_inference=false
    local need_map=false
    local need_llm=false
    local need_dashboard=false

    if [ "$force" = "true" ]; then
        need_inference=true
        need_map=true
        need_llm=true
        need_dashboard=true
    else
        if ! docker image inspect weatherscope-inference:latest &> /dev/null; then
            need_inference=true
        else
            success "Image weatherscope-inference:latest exists"
        fi

        if ! docker image inspect weatherscope-map:latest &> /dev/null; then
            need_map=true
        else
            success "Image weatherscope-map:latest exists"
        fi

        if ! docker image inspect weatherscope-llm:latest &> /dev/null; then
            need_llm=true
        else
            success "Image weatherscope-llm:latest exists"
        fi

        if ! docker image inspect weatherscope-dashboard:latest &> /dev/null; then
            need_dashboard=true
        else
            success "Image weatherscope-dashboard:latest exists"
        fi
    fi

    # Build if needed
    if [ "$need_inference" = "true" ]; then
        info "Building inference service image..."
        docker build -t weatherscope-inference:latest "$SERVICES_DIR/inference"
        success "Built weatherscope-inference:latest"
    fi

    if [ "$need_map" = "true" ]; then
        info "Building map service image..."
        docker build -t weatherscope-map:latest "$SERVICES_DIR/map"
        success "Built weatherscope-map:latest"
    fi

    if [ "$need_llm" = "true" ]; then
        info "Building LLM service image..."
        docker build -t weatherscope-llm:latest "$SERVICES_DIR/llm"
        success "Built weatherscope-llm:latest"
    fi

    if [ "$need_dashboard" = "true" ]; then
        info "Building dashboard service image..."
        docker build -t weatherscope-dashboard:latest "$SERVICES_DIR/dashboard"
        success "Built weatherscope-dashboard:latest"
    fi

    # Import to k3d if cluster exists
    if k3d cluster list 2>/dev/null | grep -q "$CLUSTER_NAME"; then
        info "Importing images to k3d cluster..."
        k3d image import weatherscope-inference:latest -c "$CLUSTER_NAME"
        k3d image import weatherscope-map:latest -c "$CLUSTER_NAME"
        k3d image import weatherscope-llm:latest -c "$CLUSTER_NAME"
        k3d image import weatherscope-dashboard:latest -c "$CLUSTER_NAME"
        success "Images imported to cluster"
    fi
}

# ============================================================================
# Deploy Services
# ============================================================================
deploy_services() {
    info "Deploying Kubernetes resources..."

    # Namespace
    if kubectl get namespace weatherscope &> /dev/null; then
        success "Namespace 'weatherscope' exists"
    else
        info "Creating namespace..."
        kubectl apply -f "$K8S_DIR/namespace.yaml"
        success "Namespace created"
    fi

    # Deploy all resources
    info "Applying deployments..."
    kubectl apply -f "$K8S_DIR/inference-deployment.yaml"
    kubectl apply -f "$K8S_DIR/map-deployment.yaml"
    kubectl apply -f "$K8S_DIR/llm-deployment.yaml"
    kubectl apply -f "$K8S_DIR/dashboard-deployment.yaml"

    local llm_backend
    llm_backend="$(grep -E '^[[:space:]]*LLM_BACKEND:' "$K8S_DIR/llm-deployment.yaml" \
        | head -n1 \
        | sed -E 's/^[^:]*:[[:space:]]*\"?([^\"[:space:]]+)\"?.*$/\1/' || true)"

    if [ -f "$K8S_DIR/ollama-deployment.yaml" ] && [ "${llm_backend}" = "ollama" ]; then
        kubectl apply -f "$K8S_DIR/ollama-deployment.yaml"
    else
        info "Skipping Ollama deployment (LLM_BACKEND=${llm_backend:-unknown})"
    fi

    kubectl apply -f "$K8S_DIR/fourcastnet-service.yaml"
    success "Deployments applied"

    reconcile_gateway
}

# ============================================================================
# Restart Deployments
# ============================================================================
restart_deployments() {
    info "Restarting deployments to pick up latest images..."

    for dep in inference-deployment map-deployment llm-deployment dashboard-deployment; do
        if kubectl get deployment "$dep" -n weatherscope &> /dev/null; then
            kubectl rollout restart "deployment/$dep" -n weatherscope
        fi
    done

    success "Deployments restarted"
}

# ============================================================================
# Gateway Reconcile
# ============================================================================
reconcile_gateway() {
    # Always reconcile gateway routes so endpoint changes are picked up
    # during normal runs (no dedicated --gateway step required).
    info "Reconciling gateway configuration..."
    kubectl apply -f "$K8S_DIR/gateway.yaml"
    success "Gateway configuration applied"
}

# ============================================================================
# Wait for Ready
# ============================================================================
wait_for_ready() {
    info "Waiting for services to be ready..."

    # Wait for gateway
    local gateway_ready=false
    for i in {1..30}; do
        if kubectl get gateway weatherscope-gateway -n weatherscope -o jsonpath='{.status.conditions[?(@.type=="Programmed")].status}' 2>/dev/null | grep -q "True"; then
            gateway_ready=true
            break
        fi
        echo -n "."
        sleep 2
    done
    echo ""

    if [ "$gateway_ready" = "true" ]; then
        success "Gateway is ready"
    else
        warn "Gateway may not be fully ready yet"
    fi

    # Wait for deployments
    kubectl wait --timeout=2m -n weatherscope deployment/inference-deployment \
        --for=condition=Available 2>/dev/null && success "Inference service ready" || warn "Inference still starting"

    kubectl wait --timeout=2m -n weatherscope deployment/map-deployment \
        --for=condition=Available 2>/dev/null && success "Map service ready" || warn "Map still starting"

    kubectl wait --timeout=2m -n weatherscope deployment/llm-deployment \
        --for=condition=Available 2>/dev/null && success "LLM service ready" || warn "LLM still starting"

    kubectl wait --timeout=2m -n weatherscope deployment/dashboard-deployment \
        --for=condition=Available 2>/dev/null && success "Dashboard service ready" || warn "Dashboard still starting"
}

# ============================================================================
# Status
# ============================================================================
print_status() {
    local endpoint_port="$K3D_LB_PORT"
    local access_mode="k3d loadbalancer"
    if [ "$PORT_FORWARD_ENABLED" = "true" ]; then
        endpoint_port="$PORT_FORWARD_PORT"
        access_mode="kubectl port-forward"
    fi

    echo ""
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}  WeatherScope Status${NC}"
    echo -e "${BLUE}================================================${NC}"

    # Cluster status
    echo -e "\n${YELLOW}Cluster:${NC}"
    if k3d cluster list 2>/dev/null | grep -q "$CLUSTER_NAME"; then
        echo "  $CLUSTER_NAME: running"
    else
        echo "  $CLUSTER_NAME: not found"
        return
    fi

    # Kgateway status
    echo -e "\n${YELLOW}Kgateway:${NC}"
    if kubectl get deployment kgateway -n kgateway-system &> /dev/null; then
        local replicas=$(kubectl get deployment kgateway -n kgateway-system -o jsonpath='{.status.availableReplicas}' 2>/dev/null)
        echo "  Controller: ${replicas:-0}/1 ready"
    else
        echo "  Controller: not installed"
    fi

    # Pods
    echo -e "\n${YELLOW}Pods:${NC}"
    kubectl get pods -n weatherscope 2>/dev/null || echo "  No pods found"

    # Services
    echo -e "\n${YELLOW}Services:${NC}"
    kubectl get svc -n weatherscope 2>/dev/null || echo "  No services found"

    # Gateway
    echo -e "\n${YELLOW}Gateway:${NC}"
    kubectl get gateway -n weatherscope 2>/dev/null || echo "  No gateway found"

    # HTTPRoutes
    echo -e "\n${YELLOW}HTTPRoutes:${NC}"
    kubectl get httproute -n weatherscope 2>/dev/null || echo "  No routes found"

    # Access info
    echo ""
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}  Access Information${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo -e "\n${GREEN}Gateway URL (${access_mode}): http://localhost:${endpoint_port}${NC}"
    if [ "$PORT_FORWARD_ENABLED" != "true" ]; then
        echo "Gateway URL (kubectl port-forward): http://localhost:${PORT_FORWARD_PORT}  (use with --port-forward)"
    else
        echo "Gateway URL (k3d loadbalancer): http://localhost:${K3D_LB_PORT}"
    fi

    echo -e "\n${YELLOW}IMPORTANT: Start FourCastNet on your Mac:${NC}"
    echo "  ./scripts/run-fourcastnet.sh"

    echo -e "\nEndpoints (via gateway):"
    echo "  - Health:     http://localhost:${endpoint_port}/health"
    echo "  - Regions:    http://localhost:${endpoint_port}/regions"
    echo "  - Forecast:   http://localhost:${endpoint_port}/forecast"
    echo "  - Downscale:  http://localhost:${endpoint_port}/downscale"
    echo "  - Map:        http://localhost:${endpoint_port}/map"
    echo "  - Map (Regional): http://localhost:${endpoint_port}/map/regional"
    echo "  - Map (Downscale): http://localhost:${endpoint_port}/map/downscale"
    echo "  - Interpret:  http://localhost:${endpoint_port}/interpret"
    echo "  - Extract Context: http://localhost:${endpoint_port}/extract-context"
    echo "  - Dashboard:  http://localhost:${endpoint_port}/dashboard"
    echo "  - Dashboard API: http://localhost:${endpoint_port}/api/regions"
    echo "  - Advisory Extract API: http://localhost:${endpoint_port}/api/advisory/extract"
    echo "  - Map Health: http://localhost:${endpoint_port}/health/map"
    echo "  - Dashboard Health: http://localhost:${endpoint_port}/health/dashboard"

    echo -e "\n${YELLOW}Test:${NC}"
    echo "  curl http://localhost:${endpoint_port}/health"
}

# ============================================================================
# Cleanup
# ============================================================================
cleanup() {
    warn "Deleting cluster '$CLUSTER_NAME'..."

    if k3d cluster list 2>/dev/null | grep -q "$CLUSTER_NAME"; then
        k3d cluster delete "$CLUSTER_NAME"
        success "Cluster deleted"
    else
        info "Cluster not found"
    fi
}

# ============================================================================
# Reset (delete namespace but keep cluster)
# ============================================================================
reset_namespace() {
    info "Resetting weatherscope namespace (keeping cluster)..."

    if kubectl get namespace weatherscope &> /dev/null; then
        kubectl delete namespace weatherscope --timeout=60s || {
            warn "Force deleting stuck resources..."
            kubectl delete namespace weatherscope --force --grace-period=0 2>/dev/null || true
        }
        success "Namespace deleted"
    else
        info "Namespace not found"
    fi

    info "To redeploy, run: ./scripts/run-k8s.sh --deploy"
}

# ============================================================================
# Help
# ============================================================================
print_help() {
    echo "WeatherScope Kubernetes Deployment"
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Setup:"
    echo "  (none)        Full setup - creates/checks cluster, gateway, builds images, deploys"
    echo "  --deploy, -d  Deploy/update K8s resources (skip cluster/gateway setup)"
    echo "  --rebuild, -r Rebuild Docker images and redeploy"
    echo "  --gateway, -g Redeploy gateway configuration only"
    echo ""
    echo "Status:"
    echo "  --status, -s  Show current deployment status"
    echo ""
    echo "Port Forward:"
    echo "  (default)                 Port-forward enabled on localhost:8088"
    echo "  --no-port-forward         Disable kubectl port-forward"
    echo "  --port-forward [port]     Enable port-forward, optional custom local port"
    echo "  --lb-port [port]          k3d load balancer host port (default: ${K3D_LB_PORT})"
    echo ""
    echo "Cleanup:"
    echo "  --reset       Delete namespace only (keep cluster for faster restart)"
    echo "  --cleanup, -c Delete the entire k3d cluster"
    echo ""
    echo "  --help, -h    Show this help"
    echo ""
    echo "After deployment, start FourCastNet:"
    echo "  ./scripts/run-fourcastnet.sh"
}

# ============================================================================
# Port Forward
# ============================================================================
start_port_forward() {
    local local_port="$1"
    local endpoint_ready=false

    info "Starting port-forward: localhost:${local_port} -> svc/weatherscope-gateway:80"
    echo -e "${YELLOW}Press Ctrl+C to stop port-forward${NC}"

    # Avoid a race where Gateway is programmed but backing pod/endpoints
    # are not ready yet.
    info "Waiting for weatherscope-gateway endpoints..."
    for i in {1..30}; do
        if kubectl get endpoints weatherscope-gateway -n weatherscope -o jsonpath='{.subsets[0].addresses[0].ip}' 2>/dev/null | grep -qE '.'; then
            endpoint_ready=true
            break
        fi
        echo -n "."
        sleep 2
    done
    echo ""

    if [ "$endpoint_ready" != "true" ]; then
        warn "Gateway service has no ready endpoints yet"
        warn "Retry later: ./scripts/run-k8s.sh --port-forward ${local_port}"
        return 1
    fi

    kubectl -n weatherscope port-forward svc/weatherscope-gateway "${local_port}:80"
}

# ============================================================================
# Main
# ============================================================================
main() {
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}  WeatherScope Kubernetes Deployment${NC}"
    echo -e "${BLUE}================================================${NC}"

    local mode="full"
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --help|-h)
                mode="help"
                shift
                ;;
            --status|-s)
                mode="status"
                shift
                ;;
            --cleanup|-c)
                mode="cleanup"
                shift
                ;;
            --reset)
                mode="reset"
                shift
                ;;
            --rebuild|-r)
                mode="rebuild"
                shift
                ;;
            --gateway|-g)
                mode="gateway"
                shift
                ;;
            --deploy|-d)
                mode="deploy"
                shift
                ;;
            --port-forward)
                PORT_FORWARD_ENABLED=true
                if [[ -n "${2:-}" && "${2:-}" != -* ]]; then
                    PORT_FORWARD_PORT="$2"
                    shift 2
                else
                    shift
                fi
                ;;
            --no-port-forward)
                PORT_FORWARD_ENABLED=false
                shift
                ;;
            --lb-port)
                if [[ -n "${2:-}" && "${2:-}" != -* ]]; then
                    K3D_LB_PORT="$2"
                    shift 2
                else
                    error "--lb-port requires a port value"
                    exit 1
                fi
                ;;
            *)
                error "Unknown option: $1"
                echo ""
                print_help
                exit 1
                ;;
        esac
    done

    if [ "$PORT_FORWARD_ENABLED" = "true" ] && [ "$PORT_FORWARD_PORT" = "$K3D_LB_PORT" ]; then
        local bumped_port=$((K3D_LB_PORT + 1))
        warn "Port-forward port ${PORT_FORWARD_PORT} matches load balancer port ${K3D_LB_PORT}; using ${bumped_port} for port-forward"
        PORT_FORWARD_PORT="${bumped_port}"
    fi

    case "$mode" in
        help)
            print_help
            ;;
        status)
            print_status
            ;;
        cleanup)
            cleanup
            ;;
        reset)
            reset_namespace
            ;;
        rebuild)
            check_prerequisites
            ensure_cluster
            ensure_kgateway
            build_images true
            deploy_services
            restart_deployments
            wait_for_ready
            print_status
            ;;
        gateway)
            info "Redeploying gateway configuration..."
            kubectl apply -f "$K8S_DIR/gateway.yaml"
            success "Gateway configuration applied"
            ;;
        deploy)
            check_prerequisites
            ensure_cluster
            ensure_kgateway
            build_images
            deploy_services
            wait_for_ready
            print_status
            ;;
        *)
            # Full setup
            check_prerequisites
            ensure_cluster
            ensure_kgateway
            build_images
            deploy_services
            wait_for_ready
            print_status
            ;;
    esac

    if [ "$PORT_FORWARD_ENABLED" = "true" ] && [[ "$mode" != "help" && "$mode" != "cleanup" && "$mode" != "reset" && "$mode" != "status" ]]; then
        start_port_forward "$PORT_FORWARD_PORT"
    fi
}

main "$@"
