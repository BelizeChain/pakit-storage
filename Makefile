.PHONY: help install dev test clean build run docker

help:
@echo "Pakit Makefile Commands:"
@echo "  make install    - Install Pakit package"
@echo "  make dev        - Install with dev dependencies"
@echo "  make test       - Run test suite"
@echo "  make clean      - Clean artifacts"
@echo "  make build      - Build distribution package"
@echo "  make run        - Start API server"
@echo "  make docker     - Build and run Docker containers"
@echo "  make format     - Format code with Black"
@echo "  make lint       - Run linters"

install:
pip install -e .

dev:
pip install -e ".[dev,ml]"

test:
pytest tests/ -v

clean:
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete
find . -type d -name ".pytest_cache" -exec rm -rf {} +
rm -rf build/ dist/ *.egg-info/
rm -rf htmlcov/ .coverage

build:
python setup.py sdist bdist_wheel

run:
python api_server.py

docker:
docker-compose up --build

format:
black pakit/ tests/

lint:
flake8 pakit/ tests/
mypy pakit/
