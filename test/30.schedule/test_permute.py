import freetensor as ft

def test_5_point_seidel():
    def schd(s: ft.Schedule):
        s.permute(['L1', 'L2'], lambda i, j: (i + j, j))
    
    @ft.schedule(callback=schd)
    @ft.transform
    def test(x: ft.Var[(8, 8), 'float32', 'inout']):
        #! nid: L1
        for i in range(1, 7):
            #! nid: L2
            for j in range(1, 7):
                x[i, j] += x[i - 1, j] + x[i, j - 1] + x[i, j + 1] + x[i + 1, j]
