# Makefile for Handwriting Recognition Project
# Automates environment setup, dependency installation, testing, and deployment

.PHONY: help check setup pre-commit-install quality clean clean-cache clean-reports clean-venv test test-coverage test-fast test-watch info

# Default Python version
PYTHON_VERSION := 3.12
PYTHON_VERSION_FUZZY := 3
PYTHON := python3
VENV := .venv
BIN := $(VENV)/bin
PIP := $(BIN)/pip
PYTEST := $(BIN)/pytest
PYTHON_VENV := $(BIN)/python
PRE_COMMIT := $(BIN)/pre-commit

# Project directories
REPORT_DIR := $(CURDIR)/reports

# Python executable paths
PYTHON_CANDIDATES := python$(PYTHON_VERSION) /usr/bin/python$(PYTHON_VERSION) /usr/local/bin/python$(PYTHON_VERSION) /opt/homebrew/bin/python$(PYTHON_VERSION) python$(PYTHON_VERSION_FUZZY) python3 python

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

# OS-specific settings
ifeq ($(OS),Windows_NT)
	VENV := .venv
	BIN := $(VENV)\Scripts
	PIP := $(BIN)\pip.exe
	PYTEST := $(BIN)\pytest.exe
	PYTHON_VENV := $(BIN)\python.exe
	PRE_COMMIT := $(BIN)\pre-commit.exe
	PYTHON_CMD := py -$(PYTHON_VERSION)
	RM_CMD = if exist "$(1)" rmdir /s /q "$(1)"
	RMFILE_CMD = if exist "$(1)" del /f /q "$(1)"
	MKDIR_CMD = if not exist "$(1)" mkdir "$(1)"
else
	PYTHON_CMD = $(call find_python)
	RM_CMD = rm -rf "$(1)"
	RMFILE_CMD = rm -f "$(1)"
	MKDIR_CMD = mkdir -p "$(1)"
endif

#=============================================================================
# Help
#=============================================================================

help: ## Show this help message
	@echo "$(BLUE)Handwriting Recognition - Available Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""

#=============================================================================
# Environment Setup
#=============================================================================

# Function to find Python version
define find_python
$(shell for python_cmd in $(PYTHON_CANDIDATES); do \
	if command -v $$python_cmd >/dev/null 2>&1; then \
		version=$$($$python_cmd -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null); \
		major=$$(echo $$version | cut -d. -f1); \
		if [ "$$version" = "$(PYTHON_VERSION)" ] || [ "$$major" = "$(PYTHON_VERSION_FUZZY)" ]; then \
			echo $$python_cmd; \
			break; \
		fi; \
	fi; \
done)
endef

check: ## Check required tools are installed
	@echo "$(BLUE)Checking required dependencies...$(NC)"
	@echo ""
	@echo "$(YELLOW)Checking Python $(PYTHON_VERSION) installation...$(NC)"

ifeq ($(OS),Windows_NT)
	@$(PYTHON_CMD) --version >NUL 2>&1 || (echo "$(RED)✗ Python $(PYTHON_VERSION) is not installed or not found$(NC)" && echo "$(YELLOW)  Please install Python $(PYTHON_VERSION) and ensure it is in PATH$(NC)" && exit /b 1)
	@echo "$(GREEN)✓ Python $(PYTHON_VERSION) found via: $(PYTHON_CMD)$(NC)"
else
	@resolved_python="$(PYTHON_CMD)"; \
	python_source="default"; \
	if command -v pyenv >/dev/null 2>&1; then \
		pyenv_available=1; \
	else \
		pyenv_available=0; \
	fi; \
	if [ -t 0 ] && [ -t 1 ]; then \
		if [ "$$pyenv_available" -eq 1 ]; then \
			printf "$(YELLOW)Use pyenv-managed Python for this run? [Y/n]: $(NC)"; \
			read -r use_pyenv; \
			case "$$use_pyenv" in \
				n|N|no|NO) ;; \
				*) python_source="pyenv"; resolved_python="pyenv exec python" ;; \
			esac; \
		else \
			echo "$(YELLOW)pyenv not found. Using default Python discovery.$(NC)"; \
		fi; \
	elif [ "$$pyenv_available" -eq 1 ]; then \
		python_source="pyenv"; \
		resolved_python="pyenv exec python"; \
	fi; \
	if ! ( [ -n "$$resolved_python" ] && $$resolved_python --version >/dev/null 2>&1 ); then \
		if [ "$$python_source" = "pyenv" ]; then \
			echo "$(YELLOW)pyenv Python is unavailable. Falling back to default Python discovery.$(NC)"; \
			resolved_python="$(PYTHON_CMD)"; \
			python_source="default"; \
		fi; \
	fi; \
	if ! ( [ -n "$$resolved_python" ] && $$resolved_python --version >/dev/null 2>&1 ); then \
		echo "$(RED)✗ Python $(PYTHON_VERSION_FUZZY).x is not installed or not found$(NC)"; \
		echo "$(YELLOW)  Please install Python $(PYTHON_VERSION_FUZZY).x and ensure it is in PATH$(NC)"; \
		exit 1; \
	else \
		echo "$(GREEN)✓ Python found via $$python_source: $$resolved_python$(NC)"; \
	fi
endif
	@echo ""
	@echo "$(BLUE)Dependency check completed.$(NC)"

setup: check ## Set up venv and install dependencies
	@echo "$(BLUE)Setting up project environment...$(NC)"
	@echo "$(YELLOW)Creating virtual environment...$(NC)"

ifeq ($(OS),Windows_NT)
	@if not exist "$(VENV)" ( $(PYTHON_CMD) -m venv "$(VENV)" && echo "$(GREEN)Virtual environment created at $(VENV)$(NC)" ) else ( echo "$(BLUE)Virtual environment already exists at $(VENV)$(NC)" )
else
	@resolved_python="$(PYTHON_CMD)"; \
	python_source="default"; \
	if command -v pyenv >/dev/null 2>&1; then \
		pyenv_available=1; \
	else \
		pyenv_available=0; \
	fi; \
	if [ -t 0 ] && [ -t 1 ]; then \
		if [ "$$pyenv_available" -eq 1 ]; then \
			printf "$(YELLOW)Use pyenv-managed Python for this run? [Y/n]: $(NC)"; \
			read -r use_pyenv; \
			case "$$use_pyenv" in \
				n|N|no|NO) ;; \
				*) python_source="pyenv"; resolved_python="pyenv exec python" ;; \
			esac; \
		else \
			echo "$(YELLOW)pyenv not found. Using default Python discovery.$(NC)"; \
		fi; \
	elif [ "$$pyenv_available" -eq 1 ]; then \
		python_source="pyenv"; \
		resolved_python="pyenv exec python"; \
	fi; \
	if ! ( [ -n "$$resolved_python" ] && $$resolved_python --version >/dev/null 2>&1 ); then \
		if [ "$$python_source" = "pyenv" ]; then \
			echo "$(YELLOW)pyenv Python is unavailable. Falling back to default Python discovery.$(NC)"; \
			resolved_python="$(PYTHON_CMD)"; \
			python_source="default"; \
		fi; \
	fi; \
	if ! ( [ -n "$$resolved_python" ] && $$resolved_python --version >/dev/null 2>&1 ); then \
		echo "$(RED)✗ Python $(PYTHON_VERSION_FUZZY).x is not installed or not found$(NC)"; \
		echo "$(YELLOW)  Please install Python $(PYTHON_VERSION_FUZZY).x and ensure it is in PATH$(NC)"; \
		exit 1; \
	fi; \
	echo "$(GREEN)Using $$python_source Python for setup: $$resolved_python$(NC)"; \
	if [ ! -d "$(VENV)" ]; then \
		$$resolved_python -m venv $(VENV); \
		echo "$(GREEN)Virtual environment created at $(VENV)$(NC)"; \
	else \
		echo "$(BLUE)Virtual environment already exists at $(VENV)$(NC)"; \
	fi
endif
	@echo "$(YELLOW)Upgrading pip in virtual environment...$(NC)"
	@$(PIP) install --upgrade pip setuptools wheel
	@echo "$(GREEN)pip upgraded successfully.$(NC)"
	@echo "$(YELLOW)Installing Python dependencies in virtual environment...$(NC)"
	@$(PIP) install -r requirements.txt
	@echo "$(GREEN)Dependencies installed in virtual environment.$(NC)"
	@echo "$(BLUE)To activate the virtual environment manually:$(NC)"
ifeq ($(OS),Windows_NT)
	@echo "$(BLUE)  $(VENV)\\Scripts\\activate$(NC)"
else
	@echo "$(BLUE)  source $(VENV)/bin/activate$(NC)"
endif
	@$(MAKE) pre-commit-install

#=============================================================================
# code quality checks
#=============================================================================

pre-commit-install: ## activate pre-commit hooks
	@echo "$(BLUE)Installing git hooks...$(NC)"
	@$(PRE_COMMIT) install --config config/.pre-commit-config.yaml
	@echo "$(GREEN)✓ pre-commit hooks installed$(NC)"

quality: ## Run all code quality checks
	@echo "$(BLUE)Running code quality checks...$(NC)"
	@$(PRE_COMMIT) run --all-files --config config/.pre-commit-config.yaml
	@echo "$(GREEN)✓ Code quality checks completed$(NC)"
#=============================================================================
# Testing
#=============================================================================

test: ## Run all tests
	@echo "$(BLUE)Running tests...$(NC)"
	@$(PYTEST) tests/ -v

test-coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	@$(PYTEST) tests/ -v --cov=src --cov-report=html --cov-report=term

test-fast: ## Run tests excluding slow tests
	@echo "$(BLUE)Running fast tests...$(NC)"
	@$(PYTEST) tests/ -v -m "not slow"

test-watch: ## Run tests in watch mode
	@echo "$(BLUE)Running tests in watch mode...$(NC)"
	@$(PYTEST) tests/ -v --looponfail

#=============================================================================
# Cleanup
#=============================================================================

clean-cache: ## Remove Python cache files
	@echo "$(BLUE)Cleaning Python cache files...$(NC)"
	@find . -type f -name '*.pyc' -delete
	@find . -type d -name '__pycache__' -delete
	@find . -type d -name '.pytest_cache' -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name '.mypy_cache' -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✓ Cache cleaned$(NC)"

clean-reports: ## Remove report artifacts
	@echo "$(BLUE)Cleaning reports...$(NC)"
	@$(call RM_CMD,$(REPORT_DIR))
	@echo "$(GREEN)✓ Reports cleaned$(NC)"

clean: clean-cache ## Remove Python artifacts and cache files
	@echo "$(BLUE)Cleaning Python artifacts...$(NC)"
	@find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	@rm -rf htmlcov/ .coverage
	@echo "$(GREEN)✓ Cleaned$(NC)"

clean-venv: ## Remove virtual environment
	@echo "$(BLUE)Removing virtual environment...$(NC)"
	@$(call RM_CMD,$(VENV))
	@echo "$(GREEN)✓ Virtual environment removed$(NC)"

#=============================================================================
# Info
#=============================================================================

info: ## Show project information
	@echo "$(BLUE)Project Information$(NC)"
	@echo "-------------------"
	@echo "Python:      $$($(PYTHON) --version)"
	@echo "Venv path:   $(VENV)"
	@echo "Pip version: $$($(PIP) --version 2>/dev/null || echo 'Not installed')"
	@echo ""
	@echo "$(BLUE)Installed packages:$(NC)"
	@$(PIP) list 2>/dev/null || echo "Virtual environment not set up. Run 'make install'"
