.PHONY: help venv conda format style black lint check coverage pypi
.DEFAULT_GOAL = help

PROJECT_NAME = bias_correction
PROJECT_DIR = bias_correction
PYTHON = python
PIP = pip
CONDA = conda
SHELL = bash


doc:
	pdoc -f --html $(PROJECT_DIR) --output-dir docs
	mv ./docs/$(PROJECT_DIR).html ./docs/index.html
	rm -rf ./docs/$(PROJECT_DIR)
	
format:
	@printf "Checking code style with black...\n"
	black --check ${PROJECT_DIR} 
	@printf "\033[1;34mBlack passes!\033[0m\n\n"

style:
	@printf "Checking code style with pylint...\n"
	pylint ${PROJECT_DIR} 
	@printf "\033[1;34mPylint passes!\033[0m\n\n"

black:  # Format code in-place using black.
	black ${PROJECT_DIR}

lint: format style  # Lint code using pydocstyle, black and pylint.
