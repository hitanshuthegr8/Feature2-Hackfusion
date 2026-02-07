#!/bin/bash

# Research Server Startup Script
# This script sets up and starts the entire research server stack

set -e

echo "=========================================="
echo "Research Server - Production Setup"
echo "=========================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check Python version
echo ""
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.10"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    print_status "Python $python_version detected"
else
    print_error "Python 3.10+ required, found $python_version"
    exit 1
fi

# Check if Docker is running
echo ""
echo "Checking Docker..."
if docker info > /dev/null 2>&1; then
    print_status "Docker is running"
else
    print_error "Docker is not running. Please start Docker first."
    exit 1
fi

# Check if Ollama is running
echo ""
echo "Checking Ollama..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    print_status "Ollama is running"
    
    # Check if phi3:mini-4k-instruct model is available
    if curl -s http://localhost:11434/api/tags | grep -q "phi3:mini"; then
        print_status "phi3:mini-4k-instruct model found"
    else
        print_warning "phi3:mini-4k-instruct model not found"
        echo "Pulling model... (this may take a few minutes)"
        ollama pull phi3:mini-4k-instruct
        print_status "Model downloaded successfully"
    fi
else
    print_error "Ollama is not running. Please start Ollama first:"
    echo "  $ ollama serve"
    exit 1
fi

# Create virtual environment if it doesn't exist
echo ""
echo "Setting up Python environment..."
if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

# Install/upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1

# Install dependencies
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt
print_status "Dependencies installed"

# Create data directories
echo ""
echo "Creating data directories..."
mkdir -p data/pdfs
mkdir -p data/grobid_json
mkdir -p data/cache
print_status "Data directories created"

# Start GROBID with Docker Compose
echo ""
echo "Starting GROBID service..."
cd docker
if docker-compose -f grobid-compose.yml ps | grep -q "Up"; then
    print_status "GROBID is already running"
else
    docker-compose -f grobid-compose.yml up -d
    print_status "GROBID started"
    
    # Wait for GROBID to be ready
    echo "Waiting for GROBID to initialize..."
    max_wait=60
    waited=0
    while ! curl -s http://localhost:8070/api/isalive > /dev/null 2>&1; do
        if [ $waited -ge $max_wait ]; then
            print_error "GROBID failed to start within ${max_wait}s"
            exit 1
        fi
        sleep 2
        waited=$((waited + 2))
        echo -n "."
    done
    echo ""
    print_status "GROBID is ready"
fi
cd ..

# Start Flask server
echo ""
echo "=========================================="
echo "Starting Flask server..."
echo "=========================================="
echo ""
echo "Server will be available at: http://localhost:5000"
echo ""
echo "Available endpoints:"
echo "  POST   /search              - Search for papers"
echo "  GET    /papers/<paper_id>   - Get parsed paper"
echo "  GET    /analysis/<paper_id> - Get extracted info"
echo "  POST   /analysis            - Process papers"
echo "  GET    /health              - Health check"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run Flask app
python app.py
