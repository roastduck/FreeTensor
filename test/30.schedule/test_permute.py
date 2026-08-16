import pytest
import freetensor as ft


@pytest.mark.parametrize("as_subprocess", [False, True])
def test_5_point_seidel(as_subprocess):

    def schd(s: ft.Schedule):
        s.permute(['L1', 'L2'], lambda i, j: (i + j, i))

    @ft.schedule(callback=schd)
    @ft.transform
    def test(x: ft.Var[(8, 8), 'float32', 'inout']):
        #! label: L1
        for i in range(1, 7):
            #! label: L2
            for j in range(1, 7):
                x[i, j] += x[i - 1, j] + x[i, j - 1] + x[i, j + 1] + x[i + 1, j]

    @ft.simplify
    @ft.transform
    def test_expected(x: ft.Var[(8, 8), 'float32', 'inout']):
        for i_plus_j in range(2, 13):
            for i in range(ft.max(i_plus_j - 6, 1),
                           ft.min(i_plus_j - 1, 6) + 1):
                j = i_plus_j - i
                x[i, j] += x[i - 1, j] + x[i, j - 1] + x[i, j + 1] + x[i + 1, j]

    assert test.body.match(test_expected.body)


@pytest.mark.parametrize("as_subprocess", [False, True])
def test_9_point_seidel(as_subprocess):

    def schd(s: ft.Schedule):
        _, inner = s.permute(['L1', 'L2'], lambda i, j: (2 * i + j, i))
        s.parallelize(inner, 'openmp', as_subprocess=as_subprocess)

    @ft.schedule(callback=schd)
    @ft.transform
    def test(x: ft.Var[(8, 8), 'float32', 'inout']):
        #! label: L1
        for i in range(1, 7):
            #! label: L2
            for j in range(1, 7):
                delta = 0
                for ii in [-1, 0, 1]:
                    for jj in [-1, 0, 1]:
                        if ii != 0 or jj != 0:
                            delta += x[i + ii, j + jj]
                x[i, j] += delta


@pytest.mark.parametrize("as_subprocess", [False, True])
def test_9_point_seidel_failed(as_subprocess):

    def schd(s: ft.Schedule):
        _, inner = s.permute(['L1', 'L2'], lambda i, j: (i + j, i))
        s.parallelize(inner, 'openmp', as_subprocess=as_subprocess)

    with pytest.raises(ft.InvalidSchedule):

        @ft.schedule(callback=schd)
        @ft.transform
        def test(x: ft.Var[(8, 8), 'float32', 'inout']):
            #! label: L1
            for i in range(1, 7):
                #! label: L2
                for j in range(1, 7):
                    delta = 0
                    for ii in [-1, 0, 1]:
                        for jj in [-1, 0, 1]:
                            if ii != 0 or jj != 0:
                                delta += x[i + ii, j + jj]
                    x[i, j] += delta
