#!/bin/bash

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' 

export PYTHONPATH=$PYTHONPATH:$(pwd)

show_help() {
    echo "Usage: ./run.sh [opcion]"
    echo ""
    echo "Opciones disponibles:"
    echo "  tests        Run Unit Tests"
    echo "  visualize    Sampler Visualization"
    echo "  smoke        Smoke Train"
    echo "  benchmark    24 networks benchmark"
    echo "  all          Execute all the extra scripts sequentially"
    echo "  help         Shows this message"
}

run_visualize() {
    echo -e "${BLUE}==> Executing: Sampler Visualization...${NC}"
    python3 extras/visualize_samplers.py
}

run_smoke() {
    echo -e "${GREEN}==> Executing: Smoke Train...${NC}"
    python3 extras/smoke_train.py
}

run_benchmark() {
    echo -e "${GREEN}==> Executing: Benchmark...${NC}"
    python3 benchmark.py
}

run_tests() {
    echo -e "${GREEN}==> Running Unit Tests...${NC}"
    python3 -m unittest discover -s test -p "test_*.py"
}

case "$1" in
    visualize)
        run_visualize
        ;;
    smoke)
        run_smoke
        ;;
    tests)
        run_tests
        ;;
    benchmark)
        run_benchmark
        ;;
    all)
        run_visualize
        run_smoke
        run_benchmark
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        if [ -z "$1" ]; then
            echo "[ERROR] An option needss to be specified."
        else
            echo "[ERROR] Invalid option: '$1'."
        fi
        show_help
        exit 1
        ;;
esac