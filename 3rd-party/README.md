All of these submodules should be kept to the latest release commit

Since there is no easy way to install ANTLR (ANTLR4 is missing from Spack), we include it in `antlr/`. `antlr/antlr4` is the ANTLR repo, where its C++ runtime is needed. `antlr/*.jar` is the ANTLR generator binary.

cppitertools can be replaced once there are enough supports from STL, maybe after C++23
