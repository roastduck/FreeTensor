import ir
import numpy as np

def test_tiling():
	target = ir.CPU()
	device = ir.Device(target)

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

	i, j = "Li", "Lj"

	s = ir.Schedule(ir.pop_ast())
	i0, i1 = s.split(i, 32)
	j0, j1 = s.split(j, 32)
	s.reorder([i0, j0, i1, j1])

	s.cache("S0", "c", "cpu")
	s.cache(i1, "a", "cpu")
	s.cache(i1, "b", "cpu")

	ast = ir.lower(s.ast(), target)
	print(ast)

	with ir.VarDef([
			("a", (256, 256), "float32", "input", "cpu"),
			("b", (256, 256), "float32", "input", "cpu"),
			("c", (256, 256), "float32", "output", "cpu")]) as (a, b, c):
		with ir.For("i.0", 0, 8) as i0:
			with ir.For("j.0", 0, 8) as j0:
				with ir.VarDef("a.r", (32, 256), "float32", ir.AccessType.Cache, "cpu") as ar:
					with ir.For("i.1.ar", 32 * i0, 32 * i0 + 32) as i1:
						with ir.For("k.ar", 0, 256) as k:
							ir.Any()
					with ir.VarDef("b.r", (256, 32), "float32", ir.AccessType.Cache, "cpu") as br:
						with ir.For("k.br", 0, 256) as k:
							with ir.For("j.1.br", 32 * j0, 32 * j0 + 32) as j1:
								ir.Any()
						with ir.For("i.1", 0, 32) as i1:
							with ir.For("j.1", 0, 32) as j1:
								with ir.VarDef("c.w", (1, 1), "float32", ir.AccessType.Cache, "cpu") as cw:
									cw[0, 0] = 0
									with ir.For("k", 0, 256) as k:
										cw[0, 0] = cw[0, 0] + ar[i1, k] * br[k, j1]
									c[i1 + 32 * i0, j1 + 32 * j0] = cw[0, 0]
	std = ir.make_reduction(ir.pop_ast())
	assert std.match(ast)

	code, params = ir.codegen(ast, target)
	print(code)
	a_np = np.random.rand(256, 256).astype("float32")
	b_np = np.random.rand(256, 256).astype("float32")
	c_np = np.zeros((256, 256), dtype="float32")
	a_arr = ir.Array(a_np, device)
	b_arr = ir.Array(b_np, device)
	c_arr = ir.Array(c_np, device)
	driver = ir.Driver(code, params, device)
	driver.set_params({"a": a_arr, "b": b_arr, "c": c_arr})
	driver.run()
	c_np = c_arr.numpy()

	c_std = a_np @ b_np
	assert np.all(np.isclose(c_np, c_std))

