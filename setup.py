from setuptools import setup
from cmake_setuptools import *

setup(name='roastduck-ir',
      description='',
      version='0.0.0.dev0',
      packages=['ir'],
      package_dir={'ir': 'python/ir'},
      ext_modules=[CMakeExtension('ffi')],
      cmdclass={'build_ext': CMakeBuildExt}
      )
