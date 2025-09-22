#!/bin/bash
# MongoDB API Service Startup Script
set -e
# Color Definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}
print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}
print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}
print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check MongoDB Service Status
check_mongodb() {
    print_info "Checking MongoDB service status..."
    if ! docker-compose ps | grep -q "mongodb.*Up"; then
        print_warning "MongoDB service not running, starting..."
        docker-compose up -d
        sleep 10  # Wait for the containers to start
        if docker-compose ps | grep -q "mongodb.*Up"; then
            print_success "MongoDB service started successfully"
        else
            print_error "MongoDB service failed to start"
            docker-compose logs mongodb
            exit 1
        fi
    else
        print_success "MongoDB service is running"
    fi
}

# Check Python Dependencies
check_dependencies() {
    print_info "Checking Python dependencies..."
    if ! python -c "import fastapi, uvicorn, pymongo, requests" 2>/dev/null; then
        print_warning "Missing Python dependencies, installing..."
        pip install -r requirements.txt
        if python -c "import fastapi, uvicorn, pymongo, requests" 2>/dev/null; then
            print_success "Python dependencies installed successfully"
        else
            print_error "Python dependencies installation failed"
            exit 1
        fi
    else
        print_success "Python dependencies are met"
    fi
}

# Start the API Service
start_api() {
    print_info "Starting MongoDB API service..."
    print_info "API Service Address: http://localhost:8001"
    print_info "API Documentation Address: http://localhost:8001/docs"
    print_info "Press Ctrl+C to stop the service"
    echo ""
    export MONGO_USER=Maximellian
    export MONGO_PASSWORD=Youbicurti$1126
    cd /Users/kevincurtis/asi-arch
    PYTHONPATH=. python -m database.mongodb_api
}

# Main function
main() {
    echo "MongoDB API Service Startup Script"
    echo "========================"
    check_mongodb
    check_dependencies
    start_api
}

main "$@"
