import freetensor as ft


def test_softmax():

    @ft.lower(verbose=1)
    @ft.transform(verbose=1)
    def test(x: ft.Var[(1024,), "float32"]):
        e = ft.exp(x)
        s = ft.reduce_sum(e, axes=[-1])
        y = e / s
        return y

    y = list(
        filter(lambda v: v.name == test.returns[0].name,
               ft.find_all_stmt(test, "<VarDef>")))[0]
    assert y.buffer.tensor.dtype == 'float32>0'
