#!/bin/bash
echo "Install poetry..."
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
source $HOME/.poetry/env
echo "Install dependencies..."
poetry update
poetry shell
cd src
'clear'
echo "Start train..."
python train.py