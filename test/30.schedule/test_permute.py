import pytest
import freetensor as ft


def test_5_point_seidel():

    def schd(s: ft.Schedule):
        outer, _ = s.permute(['L1', 'L2'], lambda i, j: (i + j, i))
        s.parallelize(outer, 'openmp')

    @ft.schedule(callback=schd)
    @ft.transform
    def test(x: ft.Var[(8, 8), 'float32', 'inout']):
        #! nid: L1
        for i in range(1, 7):
            #! nid: L2
            for j in range(1, 7):
                x[i, j] += x[i - 1, j] + x[i, j - 1] + x[i, j + 1] + x[i + 1, j]


def test_9_point_seidel():

    def schd(s: ft.Schedule):
        outer, _ = s.permute(['L1', 'L2'], lambda i, j: (2 * i + j, i))
        s.parallelize(outer, 'openmp')

    @ft.schedule(callback=schd)
    @ft.transform
    def test(x: ft.Var[(8, 8), 'float32', 'inout']):
        #! nid: L1
        for i in range(1, 7):
            #! nid: L2
            for j in range(1, 7):
                delta = 0
                for ii in [-1, 0, 1]:
                    for jj in [-1, 0, 1]:
                        if ii != 0 or jj != 0:
                            delta += x[i + ii, j + jj]
                x[i, j] += delta


def test_9_point_seidel_failed():

    def schd(s: ft.Schedule):
        outer, _ = s.permute(['L1', 'L2'], lambda i, j: (i + j, i))
        s.parallelize(outer, 'openmp')

    with pytest.raises(ft.InvalidSchedule):

        @ft.schedule(callback=schd)
        @ft.transform
        def test(x: ft.Var[(8, 8), 'float32', 'inout']):
            #! nid: L1
            for i in range(1, 7):
                #! nid: L2
                for j in range(1, 7):
                    delta = 0
                    for ii in [-1, 0, 1]:
                        for jj in [-1, 0, 1]:
                            if ii != 0 or jj != 0:
                                delta += x[i + ii, j + jj]
                    x[i, j] += delta
