Unless otherwise stated, all of these submodules should be kept to the latest release commit

We use a custom pybind11 to avoid a typing error prior to Python 3.10 (https://github.com/sphinx-doc/sphinx/issues/8084)

Since there is no easy way to install ANTLR (ANTLR4 is missing from Spack), we include it in `antlr/`. `antlr/antlr4` is the ANTLR repo, where its C++ runtime is needed. `antlr/*.jar` is the ANTLR generator binary.

range-v3 can be replaced once there are enough supports from STL, maybe after C++23
