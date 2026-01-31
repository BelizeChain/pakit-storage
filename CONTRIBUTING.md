# Contributing to Pakit

Thank you for your interest in contributing to Pakit! This document provides guidelines for contributing to the project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Workflow](#development-workflow)
4. [Testing Guidelines](#testing-guidelines)
5. [Coding Standards](#coding-standards)
6. [Pull Request Process](#pull-request-process)
7. [Issue Reporting](#issue-reporting)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please:
- Be respectful and constructive
- Avoid harassment and discrimination
- Focus on collaboration and learning
- Report unacceptable behavior to: conduct@belizechain.bz

## Getting Started

### Prerequisites

- Python 3.11+
- IPFS daemon (optional for development)
- Redis (optional for caching)
- Git

### Development Setup

```bash
# Clone repository
git clone https://github.com/BelizeChain/belizechain.git
cd belizechain/pakit

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r pakit_requirements.txt

# Install development dependencies
pip install pytest pytest-asyncio pytest-cov black flake8 mypy

# Run tests
pytest tests/ -v
```

## Development Workflow

### 1. Fork and Branch

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/YOUR_USERNAME/belizechain.git
cd belizechain/pakit

# Create feature branch
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write clean, documented code
- Follow existing code structure
- Add tests for new features
- Update documentation as needed

### 3. Commit Guidelines

Use conventional commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code formatting (no logic changes)
- `refactor`: Code restructuring
- `test`: Adding/updating tests
- `chore`: Build process, dependencies

**Examples:**
```bash
git commit -m "feat(ml): add prefetch cache eviction policy"
git commit -m "fix(p2p): resolve gossip message duplication"
git commit -m "docs(integration): update Nawal storage guide"
```

## Testing Guidelines

### Unit Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_pakit_core.py -v

# Run with coverage
pytest --cov=pakit --cov-report=html
```

### Integration Tests

```bash
# Requires IPFS daemon and Redis
ipfs daemon &
redis-server &

pytest tests/test_integration.py -v
```

### Performance Tests

```bash
# Run ML model benchmarks
pytest tests/ml/test_ml_compression.py -v --benchmark
```

### Writing Tests

- **Unit tests**: Test individual functions/classes
- **Integration tests**: Test component interactions
- **Coverage**: Aim for >80% code coverage
- **Fixtures**: Use pytest fixtures for reusable test data

Example:
```python
import pytest
from pakit.core import PakitStorageEngine

@pytest.fixture
def storage_engine():
    return PakitStorageEngine()

def test_store_retrieve(storage_engine):
    data = b"test data"
    cid = storage_engine.store(data)
    retrieved = storage_engine.retrieve(cid)
    assert retrieved == data
```

## Coding Standards

### Python Style

- **PEP 8**: Follow Python style guide
- **Type hints**: Use type annotations
- **Docstrings**: Google-style docstrings
- **Line length**: Max 100 characters
- **Imports**: Group stdlib, third-party, local

### Code Formatting

```bash
# Format code with Black
black pakit/

# Check linting
flake8 pakit/

# Type checking
mypy pakit/
```

### Documentation

- **Docstrings** for all public functions/classes
- **Inline comments** for complex logic
- **README updates** for new features
- **API docs** for public interfaces

Example:
```python
def store(
    self,
    data: bytes,
    metadata: Optional[Dict[str, Any]] = None
) -> ContentID:
    """
    Store data in Pakit with optional metadata.
    
    Args:
        data: Raw bytes to store
        metadata: Optional metadata dictionary
        
    Returns:
        Content ID for retrieval
        
    Raises:
        ValueError: If data is empty
        StorageError: If storage operation fails
        
    Example:
        >>> engine = PakitStorageEngine()
        >>> cid = engine.store(b"hello world")
        >>> print(cid.to_base58())
        'Qm...'
    """
    # Implementation...
```

## Pull Request Process

### 1. Before Submitting

- [ ] All tests pass (`pytest tests/ -v`)
- [ ] Code is formatted (`black pakit/`)
- [ ] No linting errors (`flake8 pakit/`)
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (if applicable)

### 2. Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Documentation updated
- [ ] Tests pass locally
- [ ] No new warnings introduced
```

### 3. Review Process

1. **Automated checks**: CI/CD runs tests
2. **Code review**: Maintainers review changes
3. **Feedback**: Address review comments
4. **Approval**: 2+ maintainer approvals required
5. **Merge**: Squash and merge to main

## Issue Reporting

### Bug Reports

Include:
- **Description**: What happened vs. what should happen
- **Steps to reproduce**: Detailed reproduction steps
- **Environment**: OS, Python version, dependencies
- **Logs**: Relevant error messages/stack traces
- **Expected behavior**: What you expected to see

Template:
```markdown
**Describe the bug**
Clear description of the bug

**To Reproduce**
1. Step 1
2. Step 2
3. See error

**Expected behavior**
What should happen

**Environment**
- OS: Ubuntu 22.04
- Python: 3.11.5
- Pakit version: 1.0.0

**Logs**
```
Error traceback here
```
```

### Feature Requests

Include:
- **Use case**: Why is this feature needed?
- **Proposed solution**: How should it work?
- **Alternatives**: Other approaches considered
- **Additional context**: Mockups, diagrams, examples

## Development Best Practices

### 1. Code Organization

```
pakit/
├── core/              # Core storage engine
├── p2p/               # P2P networking
├── ml/                # Machine learning
├── backends/          # Storage backends
├── blockchain/        # Blockchain integration
└── tests/             # Test suite
```

### 2. Error Handling

```python
# Use custom exceptions
class PakitError(Exception):
    """Base exception for Pakit"""
    pass

class StorageError(PakitError):
    """Storage operation failed"""
    pass

# Handle errors gracefully
try:
    result = storage.store(data)
except StorageError as e:
    logger.error(f"Storage failed: {e}")
    # Fallback or retry logic
```

### 3. Logging

```python
import logging

logger = logging.getLogger(__name__)

# Use appropriate log levels
logger.debug("Detailed debugging information")
logger.info("General informational message")
logger.warning("Warning about potential issue")
logger.error("Error occurred but recoverable")
logger.critical("Critical error, system unstable")
```

### 4. Async Code

```python
# Use async/await for I/O operations
async def fetch_data(cid: ContentID) -> bytes:
    """Async data retrieval"""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"http://ipfs/{cid}") as resp:
            return await resp.read()
```

## Performance Guidelines

- **Profile before optimizing**: Use `cProfile`, `line_profiler`
- **Async I/O**: Use async for network operations
- **Caching**: Cache expensive computations
- **Batch operations**: Process in batches when possible
- **Memory efficiency**: Use generators for large datasets

## Security Guidelines

- **Input validation**: Validate all user inputs
- **Cryptographic verification**: Verify content hashes
- **Secure dependencies**: Keep dependencies updated
- **Secrets management**: Never commit secrets
- **Error messages**: Don't leak sensitive information

## Documentation Guidelines

### Code Documentation

- **Modules**: Describe purpose and usage
- **Classes**: Describe responsibility and usage patterns
- **Functions**: Document parameters, returns, raises
- **Examples**: Provide usage examples

### External Documentation

- **README.md**: Overview, quick start, features
- **docs/**: Detailed guides and tutorials
- **API reference**: Auto-generated from docstrings
- **CHANGELOG.md**: Track all notable changes

## Community

- **GitHub Discussions**: Questions and general discussion
- **GitHub Issues**: Bug reports and feature requests
- **Email**: development@belizechain.bz

## Recognition

Contributors will be recognized in:
- CHANGELOG.md release notes
- GitHub contributors page
- Project documentation

## Questions?

If you have questions about contributing:
- Check existing documentation
- Search GitHub issues
- Open a discussion on GitHub
- Email: development@belizechain.bz

Thank you for contributing to Pakit! 🚀
