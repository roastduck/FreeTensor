from typing import Sequence, Mapping

import numpy as np

import ffi

class Driver:
	def __init__(self, src: str, params: Sequence):
		self.driver = ffi._Driver()
		self.params = params
		self.driver.buildAndLoad(src, len(self.params))

	def set_params(self, tensors: Mapping):
		name2id = dict(zip(self.params, range(len(self.params))))
		for name in tensors:
			nth = name2id[name]
			tensor = tensors[name]
			if tensor.dtype == np.float32:
				self.driver.setParamF32(nth, tensor)
			elif tensor.dtype == np.int32:
				self.driver.setParamI32(nth, tensor)
			else:
				assert False

	def run(self):
		self.driver.run()

