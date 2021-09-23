function transformer(Q, K, V, Y, w, dilation, dilation_heads, n_heads, seq_len, feat_len)
    sqrt_d = sqrt(feat_len)
    Threads.@threads for i = 1:n_heads
        for j = 1:seq_len
            dot = zeros(Float32, 2 * w + 1)
            for k = (-w):w
                if i <= dilation_heads
                    if j + k >= 1 && j + k <= seq_len
                        for p = 1:feat_len
                            dot[k + w + 1] += Q[i, j, p] * K[i, j + k, p]
                        end
                    end
                else
                    if j + k * dilation >= 1 && j + k * dilation <= seq_len
                        for p = 1:feat_len
                            dot[k + w + 1] += Q[i, j, p] * K[i, j + k * dilation, p]
                        end
                    end
                end
            end
            maxval = -Inf
            for k = 1:2 * w + 1
                maxval = max(maxval, dot[k])
            end
            expval = zeros(Float32, 2 * w + 1)
            for k = 1:2 * w + 1
                expval[k] = exp(dot[k] - maxval)
            end
            expsum = 0
            for k = 1:2 * w + 1
                expsum += expval[k]
            end
            attn = zeros(Float32, 2 * w + 1)
            for k = 1:2 * w + 1
                attn[k] = expval[k] / expsum / sqrt_d
            end

            for p = 1:feat_len
                Y[i, j, p] = 0
            end
            if i <= dilation_heads
                for k = (-w):w
                    if j + k >= 1 && j + k <= seq_len
                        for p = 1:feat_len
                            Y[i, j, p] += attn[k + w + 1] * V[i, j + k, p]
                        end
                    end
                end
            else
                for k = (-w):w
                    if j + k * dilation >= 1 && j + k * dilation <= seq_len
                        for p = 1:feat_len
                            Y[i, j, p] += attn[k + w + 1] * V[i, j + k * dilation, p]
                        end
                    end
                end
            end
        end
    end
end

n_heads = 8
seq_len = 10000
feat_len = 512
w = 32
dilation = 4  # counts from 1
dilation_heads = 2
q = rand(Float32, (n_heads, seq_len, feat_len))
k = rand(Float32, (n_heads, seq_len, feat_len))
v = rand(Float32, (n_heads, seq_len, feat_len))
y = zeros(Float32, (n_heads, seq_len, feat_len))

test_num = 1000
transformer(q, k, v, y, w, dilation, dilation_heads, n_heads, seq_len, feat_len)
time = @timed begin
    for i = 1:test_num
        transformer(q, k, v, y, w, dilation, dilation_heads, n_heads, seq_len, feat_len)
    end
end
println("Time = " * string(time.time / test_num * 1000) * " ms")

