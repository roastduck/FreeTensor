using DelimitedFiles, Printf

function transformer(Q, K, V, Y, w, dilation, dilation_heads, n_heads, seq_len, feat_len)
    sqrt_d = sqrt(feat_len)
    # for i, j = 1: n_heads, seq_len
    Threads.@threads for ij = 0:n_heads*seq_len-1
        i = div(ij, seq_len) + 1
        j = mod(ij, seq_len) + 1
        dot = zeros(Float32, 2 * w + 1)
        for k = (-w):w
            dot[k + w + 1] = 0
            kt = (i > dilation_heads ? k : k * dilation)
            if j + kt >= 1 && j + kt <= seq_len
                for p = 1:feat_len
                    dot[k + w + 1] += Q[p, j, i] * K[p, j + kt, i]
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
        expsum = zero(Float32)
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
        for k = (-w):w
            kt = (i > dilation_heads ? k : k * dilation)
            if j + kt >= 1 && j + kt <= seq_len
                for p = 1:feat_len
                    Y[p, j, i] += attn[k + w + 1] * V[p, j + kt, i]
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
# q = rand(Float32, (feat_len, seq_len, n_heads))
# k = rand(Float32, (feat_len, seq_len, n_heads))
# v = rand(Float32, (feat_len, seq_len, n_heads))
# y = zeros(Float32, (feat_len, seq_len, n_heads))
q = reshape(readdlm(open("../q.in"), Float32), (feat_len, seq_len, n_heads))
k = reshape(readdlm(open("../k.in"), Float32), (feat_len, seq_len, n_heads))
v = reshape(readdlm(open("../v.in"), Float32), (feat_len, seq_len, n_heads))
y = zeros(Float32, (feat_len, seq_len, n_heads))

warmup_num = 10
test_num = 100
for i = 1:warmup_num
    transformer(q, k, v, y, w, dilation, dilation_heads, n_heads, seq_len, feat_len)
end
time = @timed begin
    for i = 1:test_num
        transformer(q, k, v, y, w, dilation, dilation_heads, n_heads, seq_len, feat_len)
    end
end
writedlm("y.out", [@sprintf("%.10f", i) for i in reshape(y, (1, :))], ' ')
println("Time = " * string(time.time / test_num * 1000) * " ms")

