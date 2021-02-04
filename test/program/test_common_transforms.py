import ir
import numpy as np

def test_tiling():
	with ir.VarDef([
			("a", (256, 256), "float32", "input", "cpu"),
			("b", (256, 256), "float32", "input", "cpu"),
			("c", (256, 256), "float32", "output", "cpu")]) as (a, b, c):
		with ir.For("i", 0, 256, nid="Li") as i:
			with ir.For("j", 0, 256, nid="Lj") as j:
				with ir.NamedScope("S0"):
					c[i, j] = 0
					with ir.For("k", 0, 256, nid="Lk") as k:
						ir.MarkNid("S1")
						c[i, j] = c[i, j] + a[i, k] * b[k, j]

	i, j, S1 = "Li", "Lj", "S1"

	s = ir.Schedule(ir.pop_ast())
	i0, i1 = s.split(i, 32)
	j0, j1 = s.split(j, 32)
	s.reorder([i0, j0, i1, j1])

	s.cache_write("S0", "c", "cpu")

	load_a, _ = s.cache_read(S1, "a", "cpu")
	s.move_to(load_a, j0, to_begin=True)

	load_b, _ = s.cache_read(S1, "b", "cpu")
	s.move_to(load_b, j0, to_begin=True)

	ast = ir.lower(s.ast(), ir.CPU())
	print(ast)

	with ir.VarDef([
			("a", (256, 256), "float32", "input", "cpu"),
			("b", (256, 256), "float32", "input", "cpu"),
			("c", (256, 256), "float32", "output", "cpu")]) as (a, b, c):
		with ir.For("i.0", 0, 8) as i0:
			with ir.For("j.0", 0, 8) as j0:
				with ir.VarDef([
					("a.r", (32, 256, 1, 1), "float32", ir.AccessType.Cache, "cpu"),
					("b.r", (32, 256, 1, 1), "float32", ir.AccessType.Cache, "cpu")]) as (ar, br):
					with ir.For("i.1", 0, 32) as i1:
						with ir.For("k", 0, 256) as k:
							ir.Any()
					with ir.For("j.1", 0, 32) as i1:
						with ir.For("k", 0, 256) as k:
							ir.Any()
					with ir.For("i.1", 0, 32) as i1:
						with ir.For("j.1", 0, 32) as i1:
							with ir.VarDef("c.w", (1, 1), "float32", ir.AccessType.Cache, "cpu") as cw:
								cw[0, 0] = 0
								with ir.For("k", 0, 256) as k:
									ir.Any()
								ir.Any()
	assert ir.pop_ast().match(ast)

	code, params = ir.codegen(ast, ir.CPU())
	print(code)
	a_np = np.random.rand(256, 256).astype("float32")
	b_np = np.random.rand(256, 256).astype("float32")
	c_np = np.zeros((256, 256), dtype="float32")
	a_arr = ir.Array(a_np, ir.Device(ir.CPU()))
	b_arr = ir.Array(b_np, ir.Device(ir.CPU()))
	c_arr = ir.Array(c_np, ir.Device(ir.CPU()))
	driver = ir.Driver(code, params, ir.Device(ir.CPU()))
	driver.set_params({"a": a_arr, "b": b_arr, "c": c_arr})
	driver.run()
	c_np = c_arr.numpy()

	c_std = a_np @ b_np
	assert np.all(np.isclose(c_np, c_std))

