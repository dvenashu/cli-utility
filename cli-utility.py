#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI Project Generator - Creates Python projects with uv
Works on Windows and Linux
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import List
from enum import Enum

# Fix encoding issues on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


class ProjectStep(Enum):
    """Enumeration of project creation steps."""
    CHECK_UV = "check_uv"
    GET_INPUT = "get_input"
    CREATE_STRUCTURE = "create_structure"
    INITIALIZE_UV = "initialize_uv"
    INSTALL_DEPS = "install_deps"
    GENERATE_FILES = "generate_files"
    COMPLETE = "complete"


class ProjectGenerator:
    """Handles the creation and setup of Python projects."""
    
    PROJECT_TEMPLATES = {
        "1": {
            "name": "CLI Application",
            "dependencies": ["click", "rich"],
            "dev_dependencies": ["pytest", "black", "ruff"],
            "files": ["cli.py", "utils.py"],
            "description": "Command-line tool with rich output"
        },
        "2": {
            "name": "Web API (FastAPI)",
            "dependencies": ["fastapi", "uvicorn[standard]", "pydantic"],
            "dev_dependencies": ["pytest", "httpx", "black", "ruff"],
            "files": ["main.py", "models.py", "routes.py"],
            "description": "REST API with FastAPI framework"
        },
        "3": {
            "name": "Data Science",
            "dependencies": ["pandas", "numpy", "matplotlib", "jupyter"],
            "dev_dependencies": ["pytest", "black", "ruff"],
            "files": ["analysis.py", "data_loader.py"],
            "description": "Data analysis and visualization"
        },
        "4": {
            "name": "Web Scraper",
            "dependencies": ["requests", "beautifulsoup4", "lxml"],
            "dev_dependencies": ["pytest", "black", "ruff"],
            "files": ["scraper.py", "parser.py"],
            "description": "Web scraping and data extraction"
        },
        "5": {
            "name": "Discord Bot",
            "dependencies": ["discord.py", "python-dotenv"],
            "dev_dependencies": ["pytest", "black", "ruff"],
            "files": ["bot.py", "commands.py"],
            "description": "Discord bot with commands"
        },
        "6": {
            "name": "Machine Learning",
            "dependencies": ["scikit-learn", "pandas", "numpy", "matplotlib"],
            "dev_dependencies": ["pytest", "black", "ruff", "jupyter"],
            "files": ["model.py", "train.py", "predict.py"],
            "description": "ML model training and prediction"
        },
        "7": {
            "name": "Basic/Minimal",
            "dependencies": [],
            "dev_dependencies": ["pytest", "black", "ruff"],
            "files": ["main.py"],
            "description": "Minimal Python project"
        }
    }
    
    def __init__(self):
        self.project_name = ""
        self.project_path = Path()
        self.template_choice = ""
        self.python_version = ""
        self.additional_deps: List[str] = []
        self.current_step = ProjectStep.CHECK_UV
        self.step_history: List[ProjectStep] = []
        self.cleanup_needed = False
        self.install_dev_deps = True
        self.git_init = True
        self.author_name = ""
        self.author_email = ""
    
    def check_uv_installed(self) -> bool:
        """Check if uv is installed."""
        print("\nChecking for uv installation...")
        try:
            result = subprocess.run(["uv", "--version"], capture_output=True, text=True, check=False, encoding='utf-8', errors='replace')
            if result.returncode == 0:
                print(f"uv is installed: {result.stdout.strip()}")
                return True
            return False
        except FileNotFoundError:
            return False
    
    def install_uv_instructions(self):
        """Provide uv installation instructions."""
        print("\nX uv is not installed.")
        print("\nTo install uv:")
        if sys.platform == "win32":
            print("  Windows: powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\"")
        else:
            print("  Linux/macOS: curl -LsSf https://astral.sh/uv/install.sh | sh")
        print("  Or: pip install uv")
        sys.exit(1)
    
    def get_user_input(self):
        """Collect project configuration."""
        print("=" * 60)
        print("Python Project Generator with uv")
        print("=" * 60)
        
        while True:
            self.project_name = input("\nProject name: ").strip()
            if self.project_name and self.project_name.replace("-", "").replace("_", "").isalnum():
                break
            print("X Invalid name.")
        
        default_location = Path.cwd() / self.project_name
        location_input = input(f"\nProject location (press Enter for {default_location}): ").strip()
        self.project_path = Path(location_input).expanduser().resolve() if location_input else default_location
        
        if self.project_path.exists():
            overwrite = input(f"\nWarning: Directory exists. Overwrite? (y/N): ").strip().lower()
            if overwrite != 'y':
                print("X Aborted.")
                sys.exit(0)
            shutil.rmtree(self.project_path)
        
        print("\nChoose a project template:")
        for key, template in self.PROJECT_TEMPLATES.items():
            print(f"  {key}. {template['name']} - {template['description']}")
        
        while True:
            self.template_choice = input("\nChoice (1-7): ").strip()
            if self.template_choice in self.PROJECT_TEMPLATES:
                break
            print("X Invalid choice.")
        
        print("\nPython version (leave empty for default):")
        print("  Examples: 3.11, 3.12, 3.13")
        self.python_version = input("Version: ").strip()
        
        print("\nAuthor information (optional):")
        self.author_name = input("Author name: ").strip()
        if self.author_name:
            self.author_email = input("Author email: ").strip()
        
        install_dev = input("\nInstall dev dependencies (pytest, black, ruff)? (Y/n): ").strip().lower()
        self.install_dev_deps = install_dev != 'n'
        
        git_choice = input("\nInitialize git repository? (Y/n): ").strip().lower()
        self.git_init = git_choice != 'n'
        
        print("\nAdditional dependencies (comma-separated):")
        deps_input = input("Dependencies: ").strip()
        if deps_input:
            self.additional_deps = [dep.strip() for dep in deps_input.split(",") if dep.strip()]
        
        print("\n" + "=" * 60)
        print("PROJECT CONFIGURATION SUMMARY")
        print("=" * 60)
        print(f"Name: {self.project_name}")
        print(f"Location: {self.project_path}")
        print(f"Template: {self.PROJECT_TEMPLATES[self.template_choice]['name']}")
        print(f"Python: {self.python_version if self.python_version else 'default'}")
        print(f"Dev dependencies: {'Yes' if self.install_dev_deps else 'No'}")
        print(f"Git: {'Yes' if self.git_init else 'No'}")
        if self.author_name:
            print(f"Author: {self.author_name}" + (f" <{self.author_email}>" if self.author_email else ""))
        print("=" * 60)
        
        proceed = input("\nProceed? (Y/n): ").strip().lower()
        if proceed == 'n':
            print("X Aborted.")
            sys.exit(0)
    
    def create_project_structure(self):
        """Create project directories."""
        print(f"\nCreating project at {self.project_path}...")
        try:
            self.project_path.mkdir(parents=True, exist_ok=True)
            self.cleanup_needed = True
            
            src_dir = self.project_path / "src" / self.project_name.replace("-", "_")
            src_dir.mkdir(parents=True, exist_ok=True)
            
            tests_dir = self.project_path / "tests"
            tests_dir.mkdir(exist_ok=True)
            
            (self.project_path / "docs").mkdir(exist_ok=True)
            if self.template_choice == "3":
                (self.project_path / "data").mkdir(exist_ok=True)
            
            (src_dir / "__init__.py").write_text('"""Main package."""\n__version__ = "0.1.0"\n', encoding='utf-8')
            (tests_dir / "__init__.py").write_text("", encoding='utf-8')
            
            test_content = f'''"""Tests for {self.project_name}."""
import pytest
from {self.project_name.replace("-", "_")} import __version__

def test_version():
    assert __version__ == "0.1.0"
'''
            (tests_dir / "test_basic.py").write_text(test_content, encoding='utf-8')
            
            print("[OK] Project structure created")
            return True
        except Exception as e:
            print(f"[ERROR] {e}")
            return False
    
    def initialize_uv_project(self):
        """Initialize uv project."""
        print("\nInitializing uv project...")
        try:
            original_dir = Path.cwd()
            os.chdir(self.project_path)
            
            cmd = ["uv", "init", "--no-workspace", "--name", self.project_name]
            if self.python_version:
                cmd.extend(["--python", self.python_version])
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=False, encoding='utf-8', errors='replace')
            
            if result.returncode != 0:
                print(f"[ERROR] {result.stderr}")
                os.chdir(original_dir)
                return False
            
            self._update_pyproject_toml()
            os.chdir(original_dir)
            print("[OK] uv project initialized")
            return True
        except Exception as e:
            print(f"[ERROR] {e}")
            return False
    
    def _update_pyproject_toml(self):
        """Update pyproject.toml metadata."""
        try:
            pyproject_path = self.project_path / "pyproject.toml"
            if not pyproject_path.exists():
                return
            
            content = pyproject_path.read_text(encoding='utf-8')
            template = self.PROJECT_TEMPLATES[self.template_choice]
            
            if 'description = ' not in content:
                content = content.replace('[project]', f'[project]\ndescription = "{template["description"]}"')
            
            pyproject_path.write_text(content, encoding='utf-8')
        except Exception as e:
            print(f"[WARNING] Could not update pyproject.toml: {e}")
    
    def install_dependencies(self):
        """Install dependencies."""
        template = self.PROJECT_TEMPLATES[self.template_choice]
        all_deps = template["dependencies"] + self.additional_deps
        dev_deps = template.get("dev_dependencies", []) if self.install_dev_deps else []
        
        if not all_deps and not dev_deps:
            print("\n[SKIP] No dependencies")
            return True
        
        print("\nInstalling dependencies...")
        try:
            original_dir = Path.cwd()
            os.chdir(self.project_path)
            
            failed_deps = []
            
            if all_deps:
                print(f"\n  Regular: {', '.join(all_deps)}")
                for dep in all_deps:
                    print(f"    {dep}...", end=" ")
                    result = subprocess.run(["uv", "add", dep], capture_output=True, text=True, check=False, encoding='utf-8', errors='replace')
                    if result.returncode != 0:
                        print("[FAILED]")
                        failed_deps.append(dep)
                    else:
                        print("[OK]")
            
            if dev_deps:
                print(f"\n  Dev: {', '.join(dev_deps)}")
                for dep in dev_deps:
                    print(f"    {dep}...", end=" ")
                    result = subprocess.run(["uv", "add", "--dev", dep], capture_output=True, text=True, check=False, encoding='utf-8', errors='replace')
                    if result.returncode != 0:
                        print("[FAILED]")
                        failed_deps.append(dep)
                    else:
                        print("[OK]")
            
            os.chdir(original_dir)
            
            if failed_deps:
                print(f"\n[ERROR] Failed: {', '.join(failed_deps)}")
                return False
            
            print("\n[OK] All dependencies installed")
            return True
        except Exception as e:
            print(f"[ERROR] {e}")
            return False
    
    def generate_starter_files(self):
        """Generate starter files."""
        print("\nGenerating starter files...")
        try:
            src_dir = self.project_path / "src" / self.project_name.replace("-", "_")
            
            if self.template_choice == "1":
                self._gen_cli(src_dir)
            elif self.template_choice == "2":
                self._gen_fastapi(src_dir)
            elif self.template_choice == "3":
                self._gen_datascience(src_dir)
            elif self.template_choice == "4":
                self._gen_scraper(src_dir)
            elif self.template_choice == "5":
                self._gen_discord(src_dir)
            elif self.template_choice == "6":
                self._gen_ml(src_dir)
            else:
                self._gen_basic(src_dir)
            
            self._gen_readme()
            self._gen_gitignore()
            
            if self.template_choice == "5":
                self._gen_env()
            
            if self.git_init:
                self._init_git()
            
            print("[OK] Starter files generated")
            return True
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _gen_cli(self, src_dir):
        (src_dir / "cli.py").write_text('''"""CLI application."""
import click
from rich.console import Console

console = Console()

@click.group()
def cli():
    pass

@click.command()
@click.option('--name', default='World')
def hello(name):
    console.print(f"[bold green]Hello, {name}![/bold green]")

cli.add_command(hello)

if __name__ == '__main__':
    cli()
''', encoding='utf-8')
        (src_dir / "utils.py").write_text('"""Utilities."""\n\ndef format_output(data):\n    return str(data)\n', encoding='utf-8')
    
    def _gen_fastapi(self, src_dir):
        (src_dir / "main.py").write_text('''"""FastAPI app."""
from fastapi import FastAPI

app = FastAPI(title="My API")

@app.get("/")
async def root():
    return {"message": "Hello World"}
''', encoding='utf-8')
        (src_dir / "models.py").write_text('''"""Models."""
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float
''', encoding='utf-8')
        (src_dir / "routes.py").write_text('''"""Routes."""
from fastapi import APIRouter

router = APIRouter()

@router.get("/items")
async def list_items():
    return {"items": []}
''', encoding='utf-8')
    
    def _gen_datascience(self, src_dir):
        (src_dir / "analysis.py").write_text('''"""Data analysis."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_and_analyze():
    data = pd.DataFrame({
        'x': np.random.randn(100),
        'y': np.random.randn(100)
    })
    print(data.describe())
    return data

if __name__ == '__main__':
    df = load_and_analyze()
''', encoding='utf-8')
        (src_dir / "data_loader.py").write_text('''"""Data loading."""
import pandas as pd

def load_csv(filepath):
    return pd.read_csv(filepath)
''', encoding='utf-8')
    
    def _gen_scraper(self, src_dir):
        (src_dir / "scraper.py").write_text('''"""Web scraper."""
import requests
from bs4 import BeautifulSoup

class WebScraper:
    def __init__(self, base_url):
        self.base_url = base_url
        self.session = requests.Session()
    
    def fetch_page(self, url):
        response = self.session.get(url)
        response.raise_for_status()
        return response.text
    
    def parse_page(self, html):
        soup = BeautifulSoup(html, 'lxml')
        return {'title': soup.title.string if soup.title else ''}

if __name__ == '__main__':
    scraper = WebScraper('https://example.com')
    print("Scraper ready")
''', encoding='utf-8')
        (src_dir / "parser.py").write_text('''"""HTML parsing."""
from bs4 import BeautifulSoup

def extract_text(html, selector):
    soup = BeautifulSoup(html, 'lxml')
    return [el.get_text(strip=True) for el in soup.select(selector)]
''', encoding='utf-8')
    
    def _gen_discord(self, src_dir):
        (src_dir / "bot.py").write_text('''"""Discord bot."""
import discord
from discord.ext import commands
import os
from dotenv import load_dotenv

load_dotenv()

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f'{bot.user} connected!')

@bot.command(name='hello')
async def hello(ctx):
    await ctx.send(f'Hello {ctx.author.mention}!')

if __name__ == '__main__':
    bot.run(os.getenv('DISCORD_TOKEN'))
''', encoding='utf-8')
        (src_dir / "commands.py").write_text('''"""Bot commands."""
from discord.ext import commands

class CustomCommands(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
    
    @commands.command(name='ping')
    async def ping(self, ctx):
        await ctx.send('Pong!')
''', encoding='utf-8')
    
    def _gen_ml(self, src_dir):
        (src_dir / "model.py").write_text('''"""ML model."""
from sklearn.ensemble import RandomForestClassifier
import pickle

class MLModel:
    def __init__(self):
        self.model = RandomForestClassifier()
        self.trained = False
    
    def train(self, X, y):
        self.model.fit(X, y)
        self.trained = True
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
''', encoding='utf-8')
        (src_dir / "train.py").write_text('''"""Training script."""
from .model import MLModel
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def main():
    X, y = make_classification(n_samples=1000, n_features=20)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    model = MLModel()
    model.train(X_train, y_train)
    model.save('model.pkl')
    print("Model trained and saved")

if __name__ == '__main__':
    main()
''', encoding='utf-8')
        (src_dir / "predict.py").write_text('''"""Prediction script."""
from .model import MLModel
import numpy as np
import pickle

def main():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    sample = np.random.randn(1, 20)
    prediction = model.predict(sample)
    print(f"Prediction: {prediction[0]}")

if __name__ == '__main__':
    main()
''', encoding='utf-8')
    
    def _gen_basic(self, src_dir):
        (src_dir / "main.py").write_text(f'''"""Main module."""

def main():
    print("Hello from {self.project_name}!")

if __name__ == '__main__':
    main()
''', encoding='utf-8')
    
    def _gen_gitignore(self):
        (self.project_path / ".gitignore").write_text('''# Python
__pycache__/
*.py[cod]
.Python
*.egg-info/
dist/
build/

# uv
.venv/
uv.lock

# IDEs
.vscode/
.idea/

# Testing
.pytest_cache/
.coverage

# Jupyter
.ipynb_checkpoints/

# Env
.env

# OS
.DS_Store
''', encoding='utf-8')
    
    def _gen_env(self):
        (self.project_path / ".env.example").write_text('DISCORD_TOKEN=your_token_here\n', encoding='utf-8')
        (self.project_path / ".env").write_text('DISCORD_TOKEN=your_token_here\n', encoding='utf-8')
    
    def _gen_readme(self):
        template = self.PROJECT_TEMPLATES[self.template_choice]
        pkg = self.project_name.replace("-", "_")
        
        readme = f'''# {self.project_name}

{template['description']}

## Installation

```bash
uv sync
```

## Usage

```bash
uv run python -m {pkg}.main
```

## Development

```bash
uv add package-name
uv run pytest
uv run black src/
```

## Author

{self.author_name if self.author_name else 'Your Name'}
'''
        (self.project_path / "README.md").write_text(readme, encoding='utf-8')
    
    def _init_git(self):
        try:
            original_dir = Path.cwd()
            os.chdir(self.project_path)
            
            subprocess.run(["git", "--version"], capture_output=True, check=False)
            subprocess.run(["git", "init"], capture_output=True, check=False)
            subprocess.run(["git", "add", "."], capture_output=True, check=False)
            subprocess.run(["git", "commit", "-m", "Initial commit"], capture_output=True, check=False)
            
            os.chdir(original_dir)
        except:
            pass
    
    def print_summary(self):
        """Print summary."""
        print("\n" + "=" * 60)
        print("PROJECT CREATED SUCCESSFULLY!")
        print("=" * 60)
        print(f"\nProject: {self.project_name}")
        print(f"Location: {self.project_path}")
        print(f"Template: {self.PROJECT_TEMPLATES[self.template_choice]['name']}")
        
        print("\nQuick Start:")
        print(f"  cd {self.project_path}")
        if sys.platform == "win32":
            print(f"  .venv\\Scripts\\activate")
        else:
            print(f"  source .venv/bin/activate")
        
        pkg = self.project_name.replace("-", "_")
        if self.template_choice == "2":
            print(f"  uv run uvicorn {pkg}.main:app --reload")
        else:
            print(f"  uv run python -m {pkg}.main")
        
        print("\nUseful Commands:")
        print("  uv add package      # Add dependency")
        print("  uv run pytest       # Run tests")
        print("  uv run black src/   # Format code")
        print("\n" + "=" * 60 + "\n")
    
    def cleanup_partial_project(self):
        """Cleanup on error."""
        if self.cleanup_needed and self.project_path.exists():
            print(f"\nCleaning up {self.project_path}...")
            try:
                shutil.rmtree(self.project_path)
                print("[OK] Cleanup complete")
            except Exception as e:
                print(f"[WARNING] {e}")
    
    def retry_from_step(self, failed_step: ProjectStep) -> bool:
        """Ask to retry."""
        print("\n" + "=" * 60)
        print("ERROR OCCURRED")
        print("=" * 60)
        
        step_names = {
            ProjectStep.CHECK_UV: "Checking uv",
            ProjectStep.GET_INPUT: "Getting input",
            ProjectStep.CREATE_STRUCTURE: "Creating structure",
            ProjectStep.INITIALIZE_UV: "Initializing uv",
            ProjectStep.INSTALL_DEPS: "Installing dependencies",
            ProjectStep.GENERATE_FILES: "Generating files"
        }
        
        print(f"\nFailed at: {step_names.get(failed_step, 'Unknown')}")
        print("\nOptions:")
        print("  1. Retry from this step")
        print("  2. Restart from beginning")
        print("  3. Exit")
        
        while True:
            choice = input("\nChoice (1-3): ").strip()
            if choice == "1":
                return True
            elif choice == "2":
                self.cleanup_partial_project()
                self.current_step = ProjectStep.GET_INPUT
                self.step_history = []
                self.cleanup_needed = False
                return True
            elif choice == "3":
                self.cleanup_partial_project()
                return False
            print("X Invalid choice")
    
    def run(self):
        """Run generator."""
        while True:
            if self.current_step == ProjectStep.CHECK_UV:
                if not self.check_uv_installed():
                    self.install_uv_instructions()
                    return
                self.step_history.append(self.current_step)
                self.current_step = ProjectStep.GET_INPUT
            
            elif self.current_step == ProjectStep.GET_INPUT:
                try:
                    self.get_user_input()
                    self.step_history.append(self.current_step)
                    self.current_step = ProjectStep.CREATE_STRUCTURE
                except Exception as e:
                    print(f"[ERROR] {e}")
                    if not self.retry_from_step(ProjectStep.GET_INPUT):
                        return
            
            elif self.current_step == ProjectStep.CREATE_STRUCTURE:
                if self.create_project_structure():
                    self.step_history.append(self.current_step)
                    self.current_step = ProjectStep.INITIALIZE_UV
                else:
                    if not self.retry_from_step(ProjectStep.CREATE_STRUCTURE):
                        return
            
            elif self.current_step == ProjectStep.INITIALIZE_UV:
                if self.initialize_uv_project():
                    self.step_history.append(self.current_step)
                    self.current_step = ProjectStep.INSTALL_DEPS
                else:
                    if not self.retry_from_step(ProjectStep.INITIALIZE_UV):
                        return
            
            elif self.current_step == ProjectStep.INSTALL_DEPS:
                if self.install_dependencies():
                    self.step_history.append(self.current_step)
                    self.current_step = ProjectStep.GENERATE_FILES
                else:
                    if not self.retry_from_step(ProjectStep.INSTALL_DEPS):
                        return
            
            elif self.current_step == ProjectStep.GENERATE_FILES:
                if self.generate_starter_files():
                    self.step_history.append(self.current_step)
                    self.current_step = ProjectStep.COMPLETE
                else:
                    if not self.retry_from_step(ProjectStep.GENERATE_FILES):
                        return
            
            elif self.current_step == ProjectStep.COMPLETE:
                self.print_summary()
                break


def main():
    """Main entry point."""
    try:
        generator = ProjectGenerator()
        generator.run()
    except KeyboardInterrupt:
        print("\n\n[CANCELLED]")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
