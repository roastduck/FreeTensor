# `source` THIS FILE! AT FreeTensor ROOT!

# Prepare a virtual environment in /tmp. If already created, keep it.
python3 -m venv /tmp/venv-freetensor-ci

# Load the virtual environment.
source /tmp/venv-freetensor-ci/bin/activate

# Install Python dependencies. Need `-f` for PyTorch. `--upgrade` forces
# to use the required version even if already installed
pip3 install --upgrade -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

# Uninstall unneeded packages
pip3 freeze | grep -v -f requirements.txt - | xargs pip3 uninstall -y
