.PHONY: help venv conda format style black lint check coverage pypi
.DEFAULT_GOAL = help

PROJECT_NAME = bias_correction
PROJECT_DIR = bias_correction
PYTHON = python
PIP = pip
CONDA = conda
SHELL = bash


doc:
	pdoc -f --html $(PROJECT_NAME).py --output-dir docs
	mv ./docs/$(PROJECT_DIR).html ./docs/index.html
	rm -rf ./docs/$(PROJECT_DIR)
	
install:
	python setup.py install
	
pypi:
	${PYTHON} setup.py clean --all
	${PYTHON} setup.py rotate --match=.tar.gz,.whl,.egg,.zip --keep=0
	${PYTHON} setup.py sdist bdist_wheel
	twine upload --skip-existing dist/*
	
server:
	pdoc --http : .
	
format:
	@printf "Checking code style with black...\n"
	black --check ${PROJECT_NAME}.py 
	@printf "\033[1;34mBlack passes!\033[0m\n\n"

style:
	@printf "Checking code style with pylint...\n"
	pylint ${PROJECT_DIR} 
	@printf "\033[1;34mPylint passes!\033[0m\n\n"

black:  # Format code in-place using black.
	black ${PROJECT_DIR}

lint: format style  # Lint code using pydocstyle, black and pylint.

clean: 
	rm -rf ./build 
	rm -rf ./dist 
	rm -rf ./*-venv
	rm -rf ./*egg* 
	rm -rf ./__pycache__
