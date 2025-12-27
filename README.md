# Python Project Generator (with uv)

A modern, interactive command-line tool for quickly bootstrapping professional Python projects using **uv** — the extremely fast Python package installer and resolver from Astral.

This script guides you step-by-step to create a new Python project with a clean structure, pre-configured dependencies, starter code, testing setup, formatting/linting tools, and best practices.

Works seamlessly on **Windows**, **Linux**, and **macOS**.

## Features

- Fully interactive setup with clear prompts and validation
- 7 built-in project templates (see below)
- Modern `src/` layout with separate `tests/`, `docs/`, and optional `data/` folders
- Automatic `uv init` and dependency installation (`uv add`)
- Optional development dependencies: `pytest`, `black`, `ruff`
- Support for custom Python version, author info, extra dependencies
- Generates tailored starter code, comprehensive README, `.gitignore`
- Optional git repository initialization with initial commit
- Robust error handling with retry options and automatic cleanup on failure
- Full UTF-8 support on Windows

## Project Templates

| # | Template              | Key Dependencies                                    | Description                              |
|---|-----------------------|-----------------------------------------------------|------------------------------------------|
| 1 | CLI Application       | `click`, `rich`                                     | Command-line tool with rich output        |
| 2 | Web API (FastAPI)      | `fastapi`, `uvicorn[standard]`, `pydantic`           | Modern REST API with auto-docs           |
| 3 | Data Science          | `pandas`, `numpy`, `matplotlib`, `jupyter`          | Data analysis and visualization          |
| 4 | Web Scraper           | `requests`, `beautifulsoup4`, `lxml`                 | Web scraping and data extraction         |
| 5 | Discord Bot           | `discord.py`, `python-dotenv`                       | Feature-rich Discord bot with commands   |
| 6 | Machine Learning      | `scikit-learn`, `pandas`, `numpy`, `matplotlib`     | Model training and prediction            |
| 7 | Basic/Minimal         | (none)                                              | Simple starting point                    |

## Requirements

- Python 3.8 or higher
- **uv**[](https://github.com/astral-sh/uv)

### Install uv (one-time)

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Alternative (any OS)
pip install uv

#run
python cli-utility.py