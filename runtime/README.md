Files with the names like `xxx_runtime.h` are included by target programs, not by the compiler itself.

Files with the names like `xxx_context.h` contain information shared between multiple runs. They are included both by the compiler and target programs.

If you are going to run a generated program outside of the framework, include the header corresponding to your architecture.
