# `source` THIS FILE! AT FreeTensor ROOT!

set -ex

# Prepare a virtual environment in ~/.cache. (/tmp is not shared by NFS). If already created,
# keep it.
python3 -m venv ~/.cache/venv-freetensor-ci

# Load the virtual environment.
source ~/.cache/venv-freetensor-ci/bin/activate

# Install Python dependencies. Need `-f` for PyTorch. `--upgrade` forces
# to use the required version even if already installed
pip3 install --upgrade -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

# Uninstall unneeded packages
# NOTE: `sed` replaces `_` to `-` because they are equivalent in PyPI
pip3 freeze | grep -v "freetensor" | sed "s/_/-/g" | grep -v -f <(cat requirements.txt | sed "s/_/-/g") - | xargs --no-run-if-empty pip3 uninstall -y
