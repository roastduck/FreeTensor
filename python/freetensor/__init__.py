import os

from . import core
from . import libop

from .libop import *
from .core import *  # After libop. Let core.add overrides libop.add

base_dir = os.path.dirname(os.path.abspath(__file__))
runtime_dir = config.runtime_dir()
runtime_dir.append(base_dir + "/share/runtime_include")
config.set_runtime_dir(runtime_dir)
