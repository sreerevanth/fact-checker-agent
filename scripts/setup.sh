#!/usr/bin/env bash
# =============================================================================
# Local development setup script
# =============================================================================
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}🔍 Fact-Checking Agent — Setup${NC}\n"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv .venv
fi

source .venv/bin/activate

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install --upgrade pip -q
pip install -r requirements.txt -q

# Pre-download embedding model
echo -e "${YELLOW}Pre-downloading embedding model (all-MiniLM-L6-v2)...${NC}"
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); print('Model ready.')"

# Copy env file
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo -e "${YELLOW}Created .env from template — please fill in your API keys.${NC}"
fi

# Create data directories
mkdir -p data/faiss_index data/diskcache

echo -e "\n${GREEN}✅ Setup complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Edit .env and add your ANTHROPIC_API_KEY (and optionally NEWSAPI_KEY)"
echo "  2. Activate the venv:  source .venv/bin/activate"
echo "  3. Start the API:      python -m src.main serve"
echo "  4. Or check a claim:   python -m src.main check 'The Earth is flat'"
echo ""
echo "Docker alternative:"
echo "  docker-compose up --build"
