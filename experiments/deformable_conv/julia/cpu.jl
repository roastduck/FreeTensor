function conv(X, W1, W2, Y, n, c_in, c_out, h, w, k_h, k_w)
    Threads.@threads for i = 1:n
        for p = 1:h
            for q = 1:w
                row = zeros(Float32, (k_h, k_w))
                col = zeros(Float32, (k_h, k_w))
                row_int = zeros(Int, (k_h, k_w))
                col_int = zeros(Int, (k_h, k_w))
                for ro = 1:k_h
                    for so = 1:k_w
                        for ki = 1:c_in
                            for ri = 0:k_h - 1
                                for si = 0:k_w - 1
                                    if p + ri >= 1 && p + ri <= h && q + si >= 1 && q + si <= w
                                        row[ro, so] += X[i, ki, p + ri, q + si] * W1[ro, so, 1, ki, ri + 1, si + 1]
                                        col[ro, so] += X[i, ki, p + ri, q + si] * W1[ro, so, 2, ki, ri + 1, si + 1]
                                    end
                                end
                            end
                        end
                        row[ro, so] /= c_in
                        col[ro, so] /= c_in
                        row_int[ro, so] = trunc(Int, row[ro, so])
                        col_int[ro, so] = trunc(Int, col[ro, so])
                    end
                end
                pixel = zeros(Float32, (c_in, k_h, k_w))
                for ki = 1:c_in
                    for ro = 1:k_h
                        for so = 1:k_w
                            x = p + ro + row_int[ro, so]
                            y = q + so + col_int[ro, so]
                            if x >= 1 && x <= h && y >= 1 && y <= w
                                pixel[ki, ro, so] += X[i, ki, x, y] * (row[ro, so] - row_int[ro, so]) * (col[ro, so] - col_int[ro, so])
                            end
                            if x >= 1 && x <= h && y + 1 >= 1 && y + 1 <= w
                                pixel[ki, ro, so] += X[i, ki, x, y + 1] * (row[ro, so] - row_int[ro, so]) * (col_int[ro, so] + 1 - col[ro, so])
                            end
                            if x + 1 >= 1 && x + 1 <= h && y >= 1 && y <= w
                                pixel[ki, ro, so] += X[i, ki, x + 1, y] * (row_int[ro, so] + 1 - row[ro, so]) * (col[ro, so] - col_int[ro, so])
                            end
                            if x + 1 >= 1 && x + 1 <= h && y + 1 >= 1 && y + 1 <= w
                                pixel[ki, ro, so] += X[i, ki, x + 1, y + 1] * (row_int[ro, so] + 1 - row[ro, so]) * (col_int[ro, so] + 1 - col[ro, so])
                            end
                        end
                    end
                end
                for ko = 1:c_out
                    Y[i, ko, p, q] = 0
                    for ki = 1:c_in
                        for ro = 1:k_h
                            for so = 1:k_w
                                Y[i, ko, p, q] += pixel[ki, ro, so] * W2[ko, ki, ro, so]
                            end
                        end
                    end
                end
            end
        end
    end
end


n = 8
c_in = 256
c_out = 256
h = 56
w = 56
k_h = 3
k_w = 3
x = map(x -> x * 2 - 1, rand(Float32, (n, c_in, h, w)))
w1 = map(x -> x * 2 - 1, rand(Float32, (k_h, k_w, 2, c_in, k_h, k_w)))
w2 = map(x -> x * 2 - 1, rand(Float32, (c_out, c_in, k_h, k_w)))
y = zeros(Float32, (n, c_out, h, w))

test_num = 1000
conv(x, w1, w2, y, n, c_in, c_out, h, w, k_h, k_w)
time = @timed begin
    for i = 1:test_num
        conv(x, w1, w2, y, n, c_in, c_out, h, w, k_h, k_w)
    end
end
println("Time = " * string(time.time / test_num * 1000) * " ms")
