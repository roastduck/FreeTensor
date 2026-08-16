import os
import sys


def main():
    executable = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "bin",
        "freetensor-transform",
    )
    if not os.path.isfile(executable):
        raise FileNotFoundError(
            f"Packaged freetensor-transform executable is not found at "
            f"{executable}")
    os.execv(executable, [executable, *sys.argv[1:]])
