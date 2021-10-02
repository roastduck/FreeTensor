function conv(X, W1, W2, Y, n, c_in, c_out, h, w, k_h, k_w)
    # for i,p,q = 1:n,h,w
    Threads.@threads for ipq = 0:n*h*w-1
        i = div(ipq, h*w) + 1
        p = div(mod(ipq, h*w), w) + 1
        q = mod(ipq, w) + 1
        row = zeros(Float32, (k_w, k_h))
        col = zeros(Float32, (k_w, k_h))
        row_int = zeros(Int, (k_w, k_h))
        col_int = zeros(Int, (k_w, k_h))
        for ro = 1:k_h
            for so = 1:k_w
                for ki = 1:c_in
                    for ri = 0:k_h - 1
                        for si = 0:k_w - 1
                            if p + ri >= 1 && p + ri <= h && q + si >= 1 && q + si <= w
                                row[so, ro] += X[q + si, p + ri, ki, i] * W1[si + 1, ri + 1, ki, 1, so, ro]
                                col[so, ro] += X[q + si, p + ri, ki, i] * W1[si + 1, ri + 1, ki, 2, so, ro]
                            end
                        end
                    end
                end
                row[so, ro] /= c_in
                col[so, ro] /= c_in
                row_int[so, ro] = trunc(Int, row[so, ro])
                col_int[so, ro] = trunc(Int, col[so, ro])
            end
        end
        pixel = zeros(Float32, (k_w, k_h, c_in))
        for ki = 1:c_in
            for ro = 1:k_h
                for so = 1:k_w
                    x = p + ro + row_int[so, ro]
                    y = q + so + col_int[so, ro]
                    if x >= 1 && x <= h && y >= 1 && y <= w
                        pixel[so, ro, ki] += X[y, x, ki, i] * (row[so, ro] - row_int[so, ro]) * (col[so, ro] - col_int[so, ro])
                    end
                    if x >= 1 && x <= h && y + 1 >= 1 && y + 1 <= w
                        pixel[so, ro, ki] += X[y + 1, x, ki, i] * (row[so, ro] - row_int[so, ro]) * (col_int[so, ro] + 1 - col[so, ro])
                    end
                    if x + 1 >= 1 && x + 1 <= h && y >= 1 && y <= w
                        pixel[so, ro, ki] += X[y, x + 1, ki, i] * (row_int[so, ro] + 1 - row[so, ro]) * (col[so, ro] - col_int[so, ro])
                    end
                    if x + 1 >= 1 && x + 1 <= h && y + 1 >= 1 && y + 1 <= w
                        pixel[so, ro, ki] += X[y + 1, x + 1, ki, i] * (row_int[so, ro] + 1 - row[so, ro]) * (col_int[so, ro] + 1 - col[so, ro])
                    end
                end
            end
        end
        for ko = 1:c_out
            Y[q, p, ko, i] = 0
            for ki = 1:c_in
                for ro = 1:k_h
                    for so = 1:k_w
                        Y[q, p, ko, i] += pixel[so, ro, ki] * W2[so, ro, ki, ko]
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
x = map(x -> x * 2 - 1, rand(Float32, (w, h, c_in, n)))
w1 = map(x -> x * 2 - 1, rand(Float32, (k_w, k_h, c_in, 2, k_w, k_h)))
w2 = map(x -> x * 2 - 1, rand(Float32, (k_w, k_h, c_in, c_out)))
y = zeros(Float32, (w, h, c_out, n))

test_num = 100
conv(x, w1, w2, y, n, c_in, c_out, h, w, k_h, k_w)
time = @timed begin
    for i = 1:test_num
        conv(x, w1, w2, y, n, c_in, c_out, h, w, k_h, k_w)
    end
end
println("Time = " * string(time.time / test_num * 1000) * " ms")
