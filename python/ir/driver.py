import numpy as np

import ffi

class Driver:
	def __init__(self):
		self.driver = ffi._Driver()

	def build_and_load(self, src: str, nParam: int):
		self.driver.buildAndLoad(src, nParam)

	def set_param(self, nth: int, tensor):
		if tensor.dtype == np.float32:
			self.driver.setParamF32(nth, tensor)
		elif tensor.dtype == np.int32:
			self.driver.setParamI32(nth, tensor)
		else:
			assert False

	def run(self):
		self.driver.run()

