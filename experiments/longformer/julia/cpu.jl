using DelimitedFiles, Printf
using Flux, Zygote
using Einsum, IterTools

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

function dilated_attention(q, k, v, w, dilation)::Array{Float32}
    feat_len, seq_len, n_heads = size(q)
    sqrt_d = sqrt(feat_len)

    pad_k = pad_zeros(k, (0, w * dilation, 0))
    pad_v = pad_zeros(v, (0, w * dilation, 0))

    indexes = map(i -> i[2] + i[1] * dilation + i[3] * (seq_len + 2 * w * dilation), product(0:2*w, 1:seq_len, 0:n_heads-1))
    diag_k = view(reshape(pad_k, (feat_len, :)), :, indexes) # (feat_len, 2*w+1, seq_len, n_heads)
    diag_v = view(reshape(pad_v, (feat_len, :)), :, indexes)
    attn = dropdims(sum(broadcast(*,
        reshape(q, (feat_len, 1, seq_len, n_heads)), diag_k
        ), dims=1), dims=1)
    attn = softmax(attn, dims=1) / sqrt_d
    y = dropdims(sum(broadcast(*,
        reshape(attn, (1, 2 * w + 1, seq_len, n_heads)), diag_v
        ), dims=2), dims=2)
    return y
end

function operator_transformer(Q, K, V, w, dilation, dilation_heads)::Array{Float32}
    front_heads = dilated_attention(Q[:, :, begin:dilation_heads], K[:, :, begin:dilation_heads],
                                    V[:, :, begin:dilation_heads], w, dilation)
    back_heads = dilated_attention(Q[:, :, dilation_heads+1:end], K[:, :, dilation_heads+1:end],
                                   V[:, :, dilation_heads+1:end], w, 1)
    return cat(front_heads, back_heads, dims=3)
end

function main()
    if length(ARGS) != 2
        println("Usage: " * PROGRAM_FILE * "  Inf/For/Bac")
        exit(-1)
    end
    n_heads = 8
    seq_len = 10000
    feat_len = 512
    w = 32
    dilation = 4  # counts from 1
    dilation_heads = 2
    q = reshape(readdlm(open("../q.in"), Float32), (feat_len, seq_len, n_heads))
    k = reshape(readdlm(open("../k.in"), Float32), (feat_len, seq_len, n_heads))
    v = reshape(readdlm(open("../v.in"), Float32), (feat_len, seq_len, n_heads))
    y = zeros(Float32, (feat_len, seq_len, n_heads))
    d_y = reshape(readdlm(open("../d_y.in"), Float32), (feat_len, seq_len, n_heads))

    if ARGS[2] == "Inf"
        warmup_num = 10
        test_num = 100
        for i = 1:warmup_num
            transformer(q, k, v, y, w, dilation, dilation_heads, n_heads, seq_len, feat_len)
            writedlm("y.out", [@sprintf("%.10f", i) for i in reshape(Array(y), (1, :))], ' ')
            println("warmup: [" * string(i) * "/" * string(warmup_num) * "]  Done.")
        end
        time = @timed begin
            for i = 1:test_num
                transformer(Q, K, V, Y, w, dilation, dilation_heads, n_heads, seq_len, feat_len)
                println("test: [" * string(i) * "/" * string(test_num) * "]  Done.")
            end
        end
        writedlm("y.out", [@sprintf("%.10f", i) for i in reshape(Array(y), (1, :))], ' ')
        println("Inference Time = " * string(time.time / test_num * 1000) * " ms")
    elseif ARGS[2] == "For"
        warmup_num = 5
        test_num = 20
        for i = 1:warmup_num
            z, back = Zygote.pullback(
                (q, k, v) -> sum(operator_transformer(q, k, v, w, dilation, dilation_heads) .* d_y),
                q, k, v
            )
            println("warmup: [" * string(i) * "/" * string(warmup_num) * "]  Done.")
        end
        time = @timed begin
            for i = 1:test_num
                z, back = Zygote.pullback(
                    (q, k, v) -> sum(operator_transformer(q, k, v, w, dilation, dilation_heads) .* d_y),
                    q, k, v
                )
                println("test: [" * string(i) * "/" * string(test_num) * "]  Done.")
            end
        end
        println("Forward Time = " * string(time.time / test_num * 1000) * " ms")
    elseif ARGS[2] == "Bac"
        warmup_num = 5
        test_num = 20
        z, back = Zygote.pullback(
            (q, k, v) -> sum(operator_transformer(q, k, v, w, dilation, dilation_heads) .* d_y),
            q, k, v
        )
        for i = 1:warmup_num
            back(1)
            println("warmup: [" * string(i) * "/" * string(warmup_num) * "]  Done.")
        end
        time = @timed begin
            for i = 1:test_num
                back(1)
                println("test: [" * string(i) * "/" * string(test_num) * "]  Done.")
            end
        end
        writedlm("y.out", [@sprintf("%.10f", i) for i in reshape(Array(y), (1, :))], ' ')
        println("Forward Time = " * string(time.time / test_num * 1000) * " ms")
    else
        println("Usage: " * PROGRAM_FILE * "Inf/For/Bac")
        exit(-1)
    end
end
main()