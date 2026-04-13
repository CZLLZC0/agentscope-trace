# Contributing to AgentScope

Thank you for your interest in contributing!

## Development Setup

```bash
# Clone
git clone https://github.com/CZLLZC0/agentscope.git
cd agentscope

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check src/
```

## Code Style

- Use `ruff` for linting and formatting
- Max line length: 88 characters
- Use type hints everywhere
- Docstrings for all public APIs

## Pull Request Guidelines

1. **One feature or fix per PR**
2. **Tests required** — new functionality needs tests
3. **Run locally first**: `pytest tests/ -v`
4. **Update docs** if behavior changes
5. **Use conventional commits**: `feat:`, `fix:`, `docs:`, `refactor:`

## Reporting Issues

Please include:
- Python version
- AgentScope version
- Minimal reproduction case
- Full error traceback

## License

By contributing, you agree your contributions will be licensed under the MIT License.
