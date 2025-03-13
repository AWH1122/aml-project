#!/bin/bash
set -e
git config --local include.path ../.gitconfig
git config --local core.autocrlf input
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
