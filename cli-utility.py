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
from typing import List, Dict, Optional, Callable
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
        self.create_venv = True
        self.git_init = True
        self.author_name = ""
        self.author_email = ""
    
    def check_uv_installed(self) -> bool:
        """Check if uv is installed on the system."""
        print("\nChecking for uv installation...")
        try:
            result = subprocess.run(
                ["uv", "--version"],
                capture_output=True,
                text=True,
                check=False,
                encoding='utf-8',
                errors='replace'
            )
            if result.returncode == 0:
                print(f"uv is installed: {result.stdout.strip()}")
                return True
            return False
        except FileNotFoundError:
            return False
    
    def install_uv_instructions(self):
        """Provide instructions for installing uv."""
        print("\nX uv is not installed on your system.")
        print("\nTo install uv:")
        
        if sys.platform == "win32":
            print("  Windows (PowerShell):")
            print("    powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\"")
        else:
            print("  Linux/macOS:")
            print("    curl -LsSf https://astral.sh/uv/install.sh | sh")
        
        print("\nOr using pip:")
        print("  pip install uv")
        print("\nAfter installation, restart your terminal and run this script again.")
        sys.exit(1)
    
    def get_user_input(self):
        """Collect project configuration from user."""
        print("=" * 60)
        print("Python Project Generator with uv")
        print("=" * 60)
        
        # Project name
        while True:
            self.project_name = input("\nProject name: ").strip()
            if self.project_name and self.project_name.replace("-", "").replace("_", "").isalnum():
                break
            print("X Invalid name. Use only letters, numbers, hyphens, and underscores.")
        
        # Project location
        default_location = Path.cwd() / self.project_name
        location_input = input(f"\nProject location (press Enter for {default_location}): ").strip()
        
        if location_input:
            self.project_path = Path(location_input).expanduser().resolve()
        else:
            self.project_path = default_location
        
        # Check if directory exists
        if self.project_path.exists():
            overwrite = input(f"\nWarning: Directory exists. Overwrite? (y/N): ").strip().lower()
            if overwrite != 'y':
                print("X Aborted.")
                sys.exit(0)
            shutil.rmtree(self.project_path)
        
        # Project template
        print("\nChoose a project template:")
        for key, template in self.PROJECT_TEMPLATES.items():
            print(f"  {key}. {template['name']} - {template['description']}")
        
        while True:
            self.template_choice = input("\nChoice (1-7): ").strip()
            if self.template_choice in self.PROJECT_TEMPLATES:
                break
            print("X Invalid choice. Enter 1-7.")
        
        # Python version
        print("\nPython version (leave empty for default):")
        print("  Examples: 3.11, 3.12, 3.13")
        self.python_version = input("Version: ").strip()
        
        # Author information
        print("\nAuthor information (optional, for pyproject.toml):")
        self.author_name = input("Author name (press Enter to skip): ").strip()
        if self.author_name:
            self.author_email = input("Author email (press Enter to skip): ").strip()
        
        # Dev dependencies
        install_dev = input("\nInstall development dependencies (pytest, black, ruff)? (Y/n): ").strip().lower()
        self.install_dev_deps = install_dev != 'n'
        
        # Git initialization
        git_choice = input("\nInitialize git repository? (Y/n): ").strip().lower()
        self.git_init = git_choice != 'n'
        
        # Additional dependencies
        print("\nAdditional dependencies (comma-separated, or press Enter to skip):")
        deps_input = input("Dependencies: ").strip()
        if deps_input:
            self.additional_deps = [dep.strip() for dep in deps_input.split(",") if dep.strip()]
        
        # Show summary
        print("\n" + "=" * 60)
        print("PROJECT CONFIGURATION SUMMARY")
        print("=" * 60)
        print(f"Name: {self.project_name}")
        print(f"Location: {self.project_path}")
        print(f"Template: {self.PROJECT_TEMPLATES[self.template_choice]['name']}")
        print(f"Python: {self.python_version if self.python_version else 'default'}")
        print(f"Dev dependencies: {'Yes' if self.install_dev_deps else 'No'}")
        print(f"Git initialization: {'Yes' if self.git_init else 'No'}")
        if self.author_name:
            print(f"Author: {self.author_name}" + (f" <{self.author_email}>" if self.author_email else ""))
        print("=" * 60)
        
        proceed = input("\nProceed with project creation? (Y/n): ").strip().lower()
        if proceed == 'n':
            print("X Aborted.")
            sys.exit(0)
    
    def create_project_structure(self):
        """Create the project directory structure."""
        print(f"\nCreating project at {self.project_path}...")
        
        try:
            # Create main directory
            self.project_path.mkdir(parents=True, exist_ok=True)
            self.cleanup_needed = True  # Mark for cleanup if errors occur
            
            # Create source directory
            src_dir = self.project_path / "src" / self.project_name.replace("-", "_")
            src_dir.mkdir(parents=True, exist_ok=True)
            
            # Create tests directory
            tests_dir = self.project_path / "tests"
            tests_dir.mkdir(exist_ok=True)
            
            # Create additional directories
            (self.project_path / "docs").mkdir(exist_ok=True)
            (self.project_path / "data").mkdir(exist_ok=True) if self.template_choice == "3" else None
            
            # Create __init__.py files
            (src_dir / "__init__.py").write_text('"""Main package."""\n__version__ = "0.1.0"\n', encoding='utf-8')
            (tests_dir / "__init__.py").write_text("", encoding='utf-8')
            
            # Create a basic test file
            test_content = f'''"""Tests for {self.project_name}."""
import pytest
from {self.project_name.replace("-", "_")} import __version__


def test_version():
    """Test version is defined."""
    assert __version__ == "0.1.0"
'''
            (tests_dir / "test_basic.py").write_text(test_content, encoding='utf-8')
            
            print("[OK] Project structure created")
            print(f"  - Created {src_dir}")
            print(f"  - Created {tests_dir}")
            print(f"  - Created {self.project_path / 'docs'}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Error creating project structure: {e}")
            return False
    
    def initialize_uv_project(self):
        """Initialize uv project."""
        print("\nInitializing uv project...")
        
        try:
            # Change to project directory
            original_dir = Path.cwd()
            os.chdir(self.project_path)
            
            # Initialize uv project with specific Python version if provided
            cmd = ["uv", "init", "--no-workspace", "--name", self.project_name]
            if self.python_version:
                cmd.extend(["--python", self.python_version])
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=False, encoding='utf-8', errors='replace')
            
            if result.returncode != 0:
                print(f"[ERROR] Error initializing uv: {result.stderr}")
                os.chdir(original_dir)
                return False
            
            # Update pyproject.toml with additional metadata
            self._update_pyproject_toml()
            
            os.chdir(original_dir)
            print("[OK] uv project initialized")
            if self.python_version:
                print(f"  - Python version: {self.python_version}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Error initializing uv: {e}")
            try:
                os.chdir(original_dir)
            except:
                pass
            return False
    
    def _update_pyproject_toml(self):
        """Update pyproject.toml with additional metadata."""
        try:
            pyproject_path = self.project_path / "pyproject.toml"
            if not pyproject_path.exists():
                return
            
            content = pyproject_path.read_text(encoding='utf-8')
            
            # Add author information if provided
            if self.author_name:
                author_line = f'authors = ["{self.author_name}'
                if self.author_email:
                    author_line += f' <{self.author_email}>'
                author_line += '"]\n'
                
                # Insert after [project] section
                if '[project]' in content:
                    lines = content.split('\n')
                    new_lines = []
                    for i, line in enumerate(lines):
                        new_lines.append(line)
                        if line.strip() == '[project]':
                            # Add after project name and version
                            j = i + 1
                            while j < len(lines) and lines[j].strip() and not lines[j].startswith('['):
                                j += 1
                            if j - 1 > i:
                                new_lines.append(author_line.rstrip())
                    content = '\n'.join(new_lines)
            
            # Add description
            template = self.PROJECT_TEMPLATES[self.template_choice]
            if 'description = ' not in content:
                content = content.replace('[project]', f'[project]\ndescription = "{template["description"]}"')
            
            pyproject_path.write_text(content, encoding='utf-8')
            
        except Exception as e:
            print(f"[WARNING] Could not update pyproject.toml: {e}")
    
    def install_dependencies(self):
        """Install project dependencies using uv."""
        template = self.PROJECT_TEMPLATES[self.template_choice]
        all_deps = template["dependencies"] + self.additional_deps
        dev_deps = template.get("dev_dependencies", []) if self.install_dev_deps else []
        
        if not all_deps and not dev_deps:
            print("\n[SKIP] No dependencies to install")
            return True
        
        print("\nInstalling dependencies...")
        
        try:
            original_dir = Path.cwd()
            os.chdir(self.project_path)
            
            failed_deps = []
            
            # Install regular dependencies
            if all_deps:
                print(f"\n  Regular dependencies: {', '.join(all_deps)}")
                for dep in all_deps:
                    print(f"    Installing {dep}...", end=" ")
                    result = subprocess.run(
                        ["uv", "add", dep],
                        capture_output=True,
                        text=True,
                        check=False,
                        encoding='utf-8',
                        errors='replace'
                    )
                    
                    if result.returncode != 0:
                        print("[FAILED]")
                        print(f"      Error: {result.stderr.strip()}")
                        failed_deps.append(dep)
                    else:
                        print("[OK]")
            
            # Install dev dependencies
            if dev_deps:
                print(f"\n  Dev dependencies: {', '.join(dev_deps)}")
                for dep in dev_deps:
                    print(f"    Installing {dep}...", end=" ")
                    result = subprocess.run(
                        ["uv", "add", "--dev", dep],
                        capture_output=True,
                        text=True,
                        check=False,
                        encoding='utf-8',
                        errors='replace'
                    )
                    
                    if result.returncode != 0:
                        print("[FAILED]")
                        print(f"      Error: {result.stderr.strip()}")
                        failed_deps.append(dep)
                    else:
                        print("[OK]")
            
            os.chdir(original_dir)
            
            if failed_deps:
                print(f"\n[ERROR] Failed to install: {', '.join(failed_deps)}")
                return False
            
            print("\n[OK] All dependencies installed successfully")
            return True
            
        except Exception as e:
            print(f"[ERROR] Error installing dependencies: {e}")
            try:
                os.chdir(original_dir)
            except:
                pass
            return False
    
    def generate_starter_files(self):
        """Generate starter code files based on template."""
        print("\nGenerating starter files...")
        
        try:
            src_dir = self.project_path / "src" / self.project_name.replace("-", "_")
            template = self.PROJECT_TEMPLATES[self.template_choice]
            
            # Generate files based on template
            if self.template_choice == "1":  # CLI Application
                self._generate_cli_files(src_dir)
            elif self.template_choice == "2":  # FastAPI
                self._generate_fastapi_files(src_dir)
            elif self.template_choice == "3":  # Data Science
                self._generate_datascience_files(src_dir)
            elif self.template_choice == "4":  # Web Scraper
                self._generate_scraper_files(src_dir)
            elif self.template_choice == "5":  # Discord Bot
                self._generate_discord_files(src_dir)
            elif self.template_choice == "6":  # Machine Learning
                self._generate_ml_files(src_dir)
            else:  # Basic
                self._generate_basic_files(src_dir)
            
            # Generate README
            self._generate_readme()
            
            # Generate .gitignore
            self._generate_gitignore()
            
            # Generate .env.example for projects that need it
            if self.template_choice in ["5"]:  # Discord bot
                self._generate_env_example()
            
            # Initialize git if requested
            if self.git_init:
                self._initialize_git()
            
            print("[OK] Starter files generated")
            print(f"  - Created source files in {src_dir}")
            print(f"  - Created README.md")
            print(f"  - Created .gitignore")
            if self.git_init:
                print("  - Initialized git repository")
            return True
            
        except Exception as e:
            print(f"[ERROR] Error generating starter files: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _generate_cli_files(self, src_dir: Path):
        """Generate CLI application files."""
        main_content = '''"""CLI application entry point."""
import click
from rich.console import Console

console = Console()

@click.group()
def cli():
    """Main CLI application."""
    pass

@cli.command()
@click.option('--name', default='World', help='Name to greet')
def hello(name):
    """Say hello."""
    console.print(f"[bold green]Hello, {name}![/bold green]")

if __name__ == '__main__':
    cli()
'''
        (src_dir / "cli.py").write_text(main_content, encoding='utf-8')
        
        utils_content = '''"""Utility functions."""

def format_output(data):
    """Format data for output."""
    return str(data)
'''
        (src_dir / "utils.py").write_text(utils_content, encoding='utf-8')
    
    def _generate_fastapi_files(self, src_dir: Path):
        """Generate FastAPI files."""
        main_content = '''"""FastAPI application."""
from fastapi import FastAPI
from .routes import router

app = FastAPI(title="My API", version="0.1.0")
app.include_router(router)

@app.get("/")
async def root():
    return {"message": "Welcome to the API"}
'''
        (src_dir / "main.py").write_text(main_content, encoding='utf-8')
        
        models_content = '''"""Pydantic models."""
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    description: str = None
    price: float
'''
        (src_dir / "models.py").write_text(models_content, encoding='utf-8')
        
        routes_content = '''"""API routes."""
from fastapi import APIRouter
from .models import Item

router = APIRouter()

@router.get("/items")
async def list_items():
    return {"items": []}

@router.post("/items")
async def create_item(item: Item):
    return item
'''
        (src_dir / "routes.py").write_text(routes_content, encoding='utf-8')
    
    def _generate_datascience_files(self, src_dir: Path):
        """Generate data science files."""
        analysis_content = '''"""Data analysis module."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_and_analyze():
    """Load and analyze data."""
    # Sample data
    data = pd.DataFrame({
        'x': np.random.randn(100),
        'y': np.random.randn(100)
    })
    
    print(data.describe())
    return data

if __name__ == '__main__':
    df = load_and_analyze()
'''
        (src_dir / "analysis.py").write_text(analysis_content)
        
        loader_content = '''"""Data loading utilities."""
import pandas as pd

def load_csv(filepath):
    """Load data from CSV file."""
    return pd.read_csv(filepath)
'''
        (src_dir / "data_loader.py").write_text(loader_content)
    
    def _generate_scraper_files(self, src_dir: Path):
        """Generate web scraper files."""
        scraper_content = '''"""Web scraper module."""
import requests
from bs4 import BeautifulSoup
from typing import List, Dict


class WebScraper:
    """Simple web scraper."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def fetch_page(self, url: str) -> str:
        """Fetch page content."""
        response = self.session.get(url)
        response.raise_for_status()
        return response.text
    
    def parse_page(self, html: str) -> Dict:
        """Parse HTML content."""
        soup = BeautifulSoup(html, 'lxml')
        return {
            'title': soup.title.string if soup.title else '',
            'links': [a.get('href') for a in soup.find_all('a', href=True)]
        }


def main():
    """Example usage."""
    scraper = WebScraper('https://example.com')
    html = scraper.fetch_page('https://example.com')
    data = scraper.parse_page(html)
    print(f"Page title: {data['title']}")
    print(f"Found {len(data['links'])} links")


if __name__ == '__main__':
    main()
'''
        (src_dir / "scraper.py").write_text(scraper_content, encoding='utf-8')
        
        parser_content = '''"""HTML parsing utilities."""
from bs4 import BeautifulSoup
from typing import List


def extract_text(html: str, selector: str) -> List[str]:
    """Extract text from HTML using CSS selector."""
    soup = BeautifulSoup(html, 'lxml')
    elements = soup.select(selector)
    return [el.get_text(strip=True) for el in elements]


def extract_attributes(html: str, selector: str, attribute: str) -> List[str]:
    """Extract attributes from HTML elements."""
    soup = BeautifulSoup(html, 'lxml')
    elements = soup.select(selector)
    return [el.get(attribute, '') for el in elements]
'''
        (src_dir / "parser.py").write_text(parser_content, encoding='utf-8')
    
    def _generate_discord_files(self, src_dir: Path):
        """Generate Discord bot files."""
        bot_content = '''"""Discord bot main file."""
import discord
from discord.ext import commands
import os
from dotenv import load_dotenv

load_dotenv()

# Bot configuration
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)


@bot.event
async def on_ready():
    """Called when bot is ready."""
    print(f'{bot.user} has connected to Discord!')
    print(f'Bot is in {len(bot.guilds)} guilds')


@bot.command(name='hello')
async def hello(ctx):
    """Say hello."""
    await ctx.send(f'Hello {ctx.author.mention}!')


@bot.command(name='ping')
async def ping(ctx):
    """Check bot latency."""
    latency = round(bot.latency * 1000)
    await ctx.send(f'Pong! Latency: {latency}ms')


def main():
    """Run the bot."""
    token = os.getenv('DISCORD_TOKEN')
    if not token:
        print("Error: DISCORD_TOKEN not found in environment variables")
        print("Please create a .env file with your bot token")
        return
    
    bot.run(token)


if __name__ == '__main__':
    main()
'''
        (src_dir / "bot.py").write_text(bot_content, encoding='utf-8')
        
        commands_content = '''"""Custom bot commands."""
from discord.ext import commands
import random


class CustomCommands(commands.Cog):
    """Custom command cog."""
    
    def __init__(self, bot):
        self.bot = bot
    
    @commands.command(name='roll')
    async def roll_dice(self, ctx, dice: str = '1d6'):
        """Roll dice (e.g., !roll 2d6)."""
        try:
            rolls, sides = map(int, dice.split('d'))
            results = [random.randint(1, sides) for _ in range(rolls)]
            total = sum(results)
            await ctx.send(f'Rolled {dice}: {results} = {total}')
        except ValueError:
            await ctx.send('Invalid dice format. Use NdN (e.g., 2d6)')


async def setup(bot):
    """Setup function for cog."""
    await bot.add_cog(CustomCommands(bot))
'''
        (src_dir / "commands.py").write_text(commands_content, encoding='utf-8')
    
    def _generate_ml_files(self, src_dir: Path):
        """Generate machine learning files."""
        model_content = '''"""Machine learning model definition."""
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
import pickle
from pathlib import Path


class MLModel:
    """Simple ML model wrapper."""
    
    def __init__(self, model_type='random_forest'):
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.trained = False
    
    def train(self, X, y):
        """Train the model."""
        self.model.fit(X, y)
        self.trained = True
    
    def predict(self, X):
        """Make predictions."""
        if not self.trained:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        """Evaluate model performance."""
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)
        report = classification_report(y, predictions)
        return {'accuracy': accuracy, 'report': report}
    
    def save(self, filepath: str):
        """Save model to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
    
    @classmethod
    def load(cls, filepath: str):
        """Load model from file."""
        instance = cls()
        with open(filepath, 'rb') as f:
            instance.model = pickle.load(f)
        instance.trained = True
        return instance
'''
        (src_dir / "model.py").write_text(model_content, encoding='utf-8')
        
        train_content = '''"""Model training script."""
from .model import MLModel
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd


def main():
    """Train a simple model."""
    # Generate sample data
    print("Generating sample data...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    print("Training model...")
    model = MLModel()
    model.train(X_train, y_train)
    
    # Evaluate
    print("Evaluating model...")
    results = model.evaluate(X_test, y_test)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print("\\nClassification Report:")
    print(results['report'])
    
    # Save model
    model.save('model.pkl')
    print("\\nModel saved to model.pkl")


if __name__ == '__main__':
    main()
'''
        (src_dir / "train.py").write_text(train_content, encoding='utf-8')
        
        predict_content = '''"""Prediction script."""
from .model import MLModel
import numpy as np


def main():
    """Load model and make predictions."""
    # Load trained model
    print("Loading model...")
    model = MLModel.load('model.pkl')
    
    # Example prediction
    sample_data = np.random.randn(1, 20)
    prediction = model.predict(sample_data)
    
    print(f"Prediction: {prediction[0]}")


if __name__ == '__main__':
    main()
'''
        (src_dir / "predict.py").write_text(predict_content, encoding='utf-8')
        """Generate basic project files."""
        main_content = '''"""Main application module."""

def main():
    """Main function."""
    print("Hello from {}!")

if __name__ == '__main__':
    main()
'''.format(self.project_name)
        (src_dir / "main.py").write_text(main_content, encoding='utf-8')
    
    def _generate_readme(self):
        """Generate README.md file."""
        template = self.PROJECT_TEMPLATES[self.template_choice]
        package_name = self.project_name.replace("-", "_")
        
        # Build dependency list
        all_deps = template["dependencies"]
        dev_deps = template.get("dev_dependencies", []) if self.install_dev_deps else []
        
        readme_content = f'''# {self.project_name}

{template['description']}

## Features

- Modern Python project structure
- Dependency management with [uv](https://github.com/astral-sh/uv)
- Ready-to-use starter code
- Comprehensive testing setup
- Type hints and documentation

## Requirements

- Python {self.python_version if self.python_version else '3.11+'}
- uv (for dependency management)

## Installation

### Using uv (recommended)

```bash
# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate  # On Linux/macOS
.venv\\Scripts\\activate     # On Windows
```

### Manual installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Linux/macOS

# Install dependencies
pip install -e .
```

## Usage

'''
        
        # Add usage instructions based on template
        if self.template_choice == "1":  # CLI
            readme_content += f'''### Run the CLI

```bash
# Using uv
uv run python -m {package_name}.cli hello --name "World"

# Direct execution
python -m {package_name}.cli hello
```

### Available commands

- `hello`: Greet someone
'''
        
        elif self.template_choice == "2":  # FastAPI
            readme_content += f'''### Run the API server

```bash
# Using uv
uv run uvicorn {package_name}.main:app --reload

# Direct execution
uvicorn {package_name}.main:app --reload
```

The API will be available at `http://localhost:8000`

### API Documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
'''
        
        elif self.template_choice == "3":  # Data Science
            readme_content += f'''### Run analysis

```bash
# Using uv
uv run python -m {package_name}.analysis

# In Jupyter
jupyter notebook
```
'''
        
        elif self.template_choice == "4":  # Web Scraper
            readme_content += f'''### Run the scraper

```bash
# Using uv
uv run python -m {package_name}.scraper

# Direct execution
python -m {package_name}.scraper
```
'''
        
        elif self.template_choice == "5":  # Discord Bot
            readme_content += f'''### Setup

1. Create a Discord application at https://discord.com/developers/applications
2. Create a bot and get your token
3. Copy `.env.example` to `.env` and add your token:
   ```
   DISCORD_TOKEN=your_token_here
   ```

### Run the bot

```bash
# Using uv
uv run python -m {package_name}.bot

# Direct execution
python -m {package_name}.bot
```

### Available commands

- `!hello`: Greet the user
- `!ping`: Check bot latency
'''
        
        elif self.template_choice == "6":  # Machine Learning
            readme_content += f'''### Train a model

```bash
# Using uv
uv run python -m {package_name}.train

# Direct execution
python -m {package_name}.train
```

### Make predictions

```bash
uv run python -m {package_name}.predict
```
'''
        
        else:  # Basic
            readme_content += f'''### Run the application

```bash
# Using uv
uv run python -m {package_name}.main

# Direct execution
python -m {package_name}.main
```
'''
        
        readme_content += f'''
## Development

### Add dependencies

```bash
# Add a regular dependency
uv add package-name

# Add a development dependency
uv add --dev package-name
```

### Run tests

```bash
# Using uv
uv run pytest

# With coverage
uv run pytest --cov={package_name} --cov-report=html
```

### Code formatting and linting

```bash
# Format code with black
uv run black src/

# Lint with ruff
uv run ruff check src/
```

## Project Structure

```
{self.project_name}/
├── src/
│   └── {package_name}/
│       ├── __init__.py
'''
        
        # Add template-specific files to structure
        for file in template["files"]:
            readme_content += f"│       ├── {file}\n"
        
        readme_content += f'''├── tests/
│   ├── __init__.py
│   └── test_basic.py
├── docs/
├── .gitignore
├── pyproject.toml
└── README.md
```

## Dependencies

'''
        
        if all_deps:
            readme_content += "### Main\n\n"
            for dep in all_deps:
                readme_content += f"- {dep}\n"
            readme_content += "\n"
        
        if dev_deps:
            readme_content += "### Development\n\n"
            for dep in dev_deps:
                readme_content += f"- {dep}\n"
            readme_content += "\n"
        
        readme_content += f'''
## License

MIT License

## Author

{self.author_name if self.author_name else 'Your Name'}
'''
        
        if self.author_email:
            readme_content += f"{self.author_email}\n"
        
        (self.project_path / "README.md").write_text(readme_content, encoding='utf-8')
    
    def _generate_env_example(self):
        """Generate .env.example file for projects that need environment variables."""
        if self.template_choice == "5":  # Discord bot
            env_content = '''# Discord Bot Configuration
DISCORD_TOKEN=your_bot_token_here
'''
            (self.project_path / ".env.example").write_text(env_content, encoding='utf-8')
            (self.project_path / ".env").write_text(env_content, encoding='utf-8')
    
    def _initialize_git(self):
        """Initialize git repository."""
        try:
            original_dir = Path.cwd()
            os.chdir(self.project_path)
            
            # Check if git is available
            result = subprocess.run(
                ["git", "--version"],
                capture_output=True,
                check=False,
                encoding='utf-8',
                errors='replace'
            )
            
            if result.returncode != 0:
                print("[WARNING] git not found, skipping git initialization")
                os.chdir(original_dir)
                return
            
            # Initialize git
            subprocess.run(["git", "init"], capture_output=True, check=False)
            subprocess.run(["git", "add", "."], capture_output=True, check=False)
            subprocess.run(
                ["git", "commit", "-m", "Initial commit"],
                capture_output=True,
                check=False
            )
            
            os.chdir(original_dir)
            
        except Exception as e:
            print(f"[WARNING] Could not initialize git: {e}")
            try:
                os.chdir(original_dir)
            except:
                pass
        """Generate .gitignore file."""
        gitignore_content = '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
ENV/

# uv
.venv/
uv.lock

# IDEs
.vscode/
.idea/
*.swp
*.swo

# Testing
.pytest_cache/
.coverage
htmlcov/

# OS
.DS_Store
Thumbs.db
'''
        (self.project_path / ".gitignore").write_text(gitignore_content, encoding='utf-8')
    
    def print_summary(self):
        """Print project creation summary."""
        print("\n" + "=" * 60)
        print("PROJECT CREATED SUCCESSFULLY!")
        print("=" * 60)
        
        template = self.PROJECT_TEMPLATES[self.template_choice]
        package_name = self.project_name.replace("-", "_")
        
        print(f"\nProject: {self.project_name}")
        print(f"Location: {self.project_path}")
        print(f"Template: {template['name']}")
        print(f"Python: {self.python_version if self.python_version else 'default'}")
        
        print("\n" + "=" * 60)
        print("QUICK START")
        print("=" * 60)
        
        print(f"\n1. Navigate to project:")
        print(f"   cd {self.project_path}")
        
        print(f"\n2. Activate environment:")
        if sys.platform == "win32":
            print(f"   .venv\\Scripts\\activate")
        else:
            print(f"   source .venv/bin/activate")
        
        print(f"\n3. Run your project:")
        
        if self.template_choice == "1":  # CLI
            print(f"   uv run python -m {package_name}.cli hello")
        elif self.template_choice == "2":  # FastAPI
            print(f"   uv run uvicorn {package_name}.main:app --reload")
            print(f"   Then visit: http://localhost:8000/docs")
        elif self.template_choice == "3":  # Data Science
            print(f"   uv run python -m {package_name}.analysis")
            print(f"   Or: jupyter notebook")
        elif self.template_choice == "4":  # Web Scraper
            print(f"   uv run python -m {package_name}.scraper")
        elif self.template_choice == "5":  # Discord Bot
            print(f"   # First, add your token to .env file")
            print(f"   uv run python -m {package_name}.bot")
        elif self.template_choice == "6":  # ML
            print(f"   uv run python -m {package_name}.train")
        else:  # Basic
            print(f"   uv run python -m {package_name}.main")
        
        print(f"\n4. Run tests:")
        print(f"   uv run pytest")
        
        print("\n" + "=" * 60)
        print("USEFUL COMMANDS")
        print("=" * 60)
        
        print("\n  Add dependency:        uv add package-name")
        print("  Add dev dependency:    uv add --dev package-name")
        print("  Sync dependencies:     uv sync")
        print("  Format code:           uv run black src/")
        print("  Lint code:             uv run ruff check src/")
        print("  Run tests:             uv run pytest")
        print("  Test with coverage:    uv run pytest --cov")
        
        print("\n" + "=" * 60)
        print("DOCUMENTATION")
        print("=" * 60)
        
        print(f"\n  Project README:  {self.project_path / 'README.md'}")
        print(f"  uv docs:         https://docs.astral.sh/uv/")
        
        if self.template_choice == "2":
            print(f"  FastAPI docs:    https://fastapi.tiangolo.com/")
        elif self.template_choice == "5":
            print(f"  discord.py:      https://discordpy.readthedocs.io/")
        
        print("\n" + "=" * 60)
        print("Happy coding! " + ":)")
        print("=" * 60 + "\n")
    
    def cleanup_partial_project(self):
        """Clean up partially created project on error."""
        if self.cleanup_needed and self.project_path.exists():
            print(f"\nCleaning up partial project at {self.project_path}...")
            try:
                shutil.rmtree(self.project_path)
                print("[OK] Cleanup complete")
            except Exception as e:
                print(f"[WARNING] Could not fully clean up: {e}")
                print(f"   Please manually remove: {self.project_path}")
    
    def retry_from_step(self, failed_step: ProjectStep) -> bool:
        """Ask user if they want to retry from the failed step."""
        print("\n" + "=" * 60)
        print("ERROR OCCURRED")
        print("=" * 60)
        
        step_names = {
            ProjectStep.CHECK_UV: "Checking uv installation",
            ProjectStep.GET_INPUT: "Getting user input",
            ProjectStep.CREATE_STRUCTURE: "Creating project structure",
            ProjectStep.INITIALIZE_UV: "Initializing uv project",
            ProjectStep.INSTALL_DEPS: "Installing dependencies",
            ProjectStep.GENERATE_FILES: "Generating starter files"
        }
        
        print(f"\nFailed at step: {step_names.get(failed_step, 'Unknown')}")
        print("\nOptions:")
        print("  1. Retry from this step")
        print("  2. Restart from beginning")
        print("  3. Exit")
        
        while True:
            choice = input("\nChoice (1-3): ").strip()
            
            if choice == "1":
                return True
            elif choice == "2":
                # Reset to beginning
                self.cleanup_partial_project()
                self.current_step = ProjectStep.GET_INPUT
                self.step_history = []
                self.cleanup_needed = False
                return True
            elif choice == "3":
                self.cleanup_partial_project()
                return False
            else:
                print("X Invalid choice. Enter 1, 2, or 3.")
    
    def run(self):
        """Run the project generator with error handling and retry logic."""
        
        while True:
            # Execute steps based on current position
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
                    print(f"❌ Error getting user input: {e}")
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
                # All steps completed successfully
                self.print_summary()
                break


def main():
    """Main entry point."""
    try:
        generator = ProjectGenerator()
        generator.run()
    except KeyboardInterrupt:
        print("\n\n[CANCELLED] Cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()