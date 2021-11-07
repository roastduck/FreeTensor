function transformer(Q, K, V, Y, w, dilation, dilation_heads, n_heads, seq_len, feat_len)
    sqrt_d = sqrt(feat_len)
    # for i, j = 1: n_heads, seq_len
    Threads.@threads for ij = 0:n_heads*seq_len-1
        i = div(ij, seq_len) + 1
        j = mod(ij, seq_len) + 1
        dot = zeros(Float32, 2 * w + 1)
        for k = (-w):w
            if i <= dilation_heads
                if j + k >= 1 && j + k <= seq_len
                    for p = 1:feat_len
                        dot[k + w + 1] += Q[p, j, i] * K[p, j + k, i]
                    end
                end
            else
                if j + k * dilation >= 1 && j + k * dilation <= seq_len
                    for p = 1:feat_len
                        dot[k + w + 1] += Q[p, j, i] * K[p, j + k * dilation, i]
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
            Y[p, j, i] = 0
        end
        if i <= dilation_heads
            for k = (-w):w
                if j + k >= 1 && j + k <= seq_len
                    for p = 1:feat_len
                        Y[p, j, i] += attn[k + w + 1] * V[p, j + k, i]
                    end
                end
            end
        else
            for k = (-w):w
                if j + k * dilation >= 1 && j + k * dilation <= seq_len
                    for p = 1:feat_len
                        Y[p, j, i] += attn[k + w + 1] * V[p, j + k * dilation, i]
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
q = rand(Float32, (feat_len, seq_len, n_heads))
k = rand(Float32, (feat_len, seq_len, n_heads))
v = rand(Float32, (feat_len, seq_len, n_heads))
y = zeros(Float32, (feat_len, seq_len, n_heads))

test_num = 5
transformer(q, k, v, y, w, dilation, dilation_heads, n_heads, seq_len, feat_len)
time = @timed begin
    for i = 1:test_num
        transformer(q, k, v, y, w, dilation, dilation_heads, n_heads, seq_len, feat_len)
    end
end
println("Time = " * string(time.time / test_num * 1000) * " ms")

