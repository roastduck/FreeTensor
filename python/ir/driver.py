from typing import Sequence, Mapping

import numpy as np

import ffi

from ffi import Array

class Driver:
	def __init__(self, src: str, params: Sequence):
		self.driver = ffi._Driver()
		self.params = params
		self.driver.buildAndLoad(src, len(self.params))

	def set_params(self, tensors: Mapping):
		name2id = dict(zip(self.params, range(len(self.params))))
		for name in tensors:
			self.driver.setParam(name2id[name], tensors[name])

	def run(self):
		self.driver.run()

