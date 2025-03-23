#!/bin/bash
# Script to run ReelGenius tests

# Set color variables
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${YELLOW}***************************************"
    echo -e "* $1"
    echo -e "***************************************${NC}\n"
}

# Display help
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    print_header "ReelGenius Test Runner"
    echo "Usage: ./run_tests.sh [options]"
    echo ""
    echo "Options:"
    echo "  --unit           Run only unit tests"
    echo "  --integration    Run only integration tests (requires external services)"
    echo "  --all            Run all tests including integration tests"
    echo "  --coverage       Generate test coverage report"
    echo "  --verbose        Show verbose output"
    echo "  -h, --help       Show this help message"
    echo ""
    exit 0
fi

# Process arguments
RUN_UNIT=true
RUN_INTEGRATION=false
COVERAGE=false
VERBOSE=""

for arg in "$@"; do
    case $arg in
        --unit)
            RUN_UNIT=true
            RUN_INTEGRATION=false
            ;;
        --integration)
            RUN_UNIT=false
            RUN_INTEGRATION=true
            ;;
        --all)
            RUN_UNIT=true
            RUN_INTEGRATION=true
            ;;
        --coverage)
            COVERAGE=true
            ;;
        --verbose)
            VERBOSE="-v"
            ;;
    esac
done

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_header "Creating virtual environment"
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install required packages
print_header "Installing test dependencies"
pip install pytest pytest-cov

# Build command
PYTEST_CMD="pytest $VERBOSE"

if [ "$COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=video_generator --cov-report=term"
fi

# Run unit tests
if [ "$RUN_UNIT" = true ]; then
    print_header "Running unit tests"
    $PYTEST_CMD tests/unit
fi

# Run integration tests
if [ "$RUN_INTEGRATION" = true ]; then
    print_header "Running integration tests"
    export RUN_INTEGRATION_TESTS=true
    $PYTEST_CMD tests/integration
fi

# Generate coverage report if requested
if [ "$COVERAGE" = true ]; then
    print_header "Generating HTML coverage report"
    pytest --cov=video_generator --cov-report=html tests/
    echo -e "${GREEN}Coverage report generated in htmlcov/ directory${NC}"
fi

# Deactivate virtual environment
deactivate

print_header "Tests completed"