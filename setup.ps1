git config --local include.path ..\.gitconfig
git config --local core.autocrlf true
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
