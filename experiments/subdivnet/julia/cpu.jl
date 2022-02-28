using Printf
using Zygote

include("../../common/julia/io.jl")

function mult_conv(adj, x, w0, w1, w2, w3, n_faces, in_feats, out_feats)::Matrix{Float32}
    y = zeros(Float32, (out_feats, n_faces))
    Threads.@threads for i = 1:n_faces
        sum1 = zeros(Float32, in_feats)
        sum2 = zeros(Float32, in_feats)
        sum3 = zeros(Float32, in_feats)
        for p = 1:3
            for k = 1:in_feats
                sum1[k] += x[k, adj[p, i]]
                sum2[k] += abs(x[k, adj[p, i]] - x[k, adj[p % 3 + 1, i]])
                sum3[k] += abs(x[k, adj[p, i]] - x[k, i])
            end
        end
        # y[:, i] = (x[:, i]' * w0 + sum1' * w1 + sum2' * w2 + sum3' * w3)
        for j = 1:out_feats, k = 1:in_feats
            y[j, i] += x[k, i] * w0[k, j] + sum1[k] * w1[k, j] + sum2[k] * w2[k, j] + sum3[k] * w3[k, j]
        end
    end
    return y
end

function singl_conv(adj, x, w0, w1, w2, w3, n_faces, in_feats, out_feats)::Matrix{Float32}
    y = zeros(Float32, (out_feats, n_faces))
    for i = 1:n_faces
        sum1 = zeros(Float32, in_feats)
        sum2 = zeros(Float32, in_feats)
        sum3 = zeros(Float32, in_feats)
        for p = 1:3
            for k = 1:in_feats
                sum1[k] += x[k, adj[p, i]]
                sum2[k] += abs(x[k, adj[p, i]] - x[k, adj[p % 3 + 1, i]])
                sum3[k] += abs(x[k, adj[p, i]] - x[k, i])
            end
        end
        y[:, i] = (x[:, i]' * w0 + sum1' * w1 + sum2' * w2 + sum3' * w3)
    end
    return y
end

function operator_conv(adj, x, w0, w1, w2, w3, n_faces, in_feats, out_feats)::Matrix{Float32}
    adj_feat = reshape(x[:, reshape(adj, :)], (in_feats, n_faces, 3))   # (in_feats, n_faces, 3)
    y = w0 * x
    y += w1 * dropdims(sum(adj_feat, dims=3), dims=3)
    y += w2 * dropdims(sum(abs.(adj_feat .- cat(adj_feat[:, :, 2:3], reshape(adj_feat[:, :, 1], (in_feats, n_faces, 1)), dims=3)), dims=3), dims=3)
    y += w3 * dropdims(sum(abs.(broadcast(-, adj_feat, reshape(x, (in_feats, n_faces, 1)))), dims=3), dims=3)
    return y
end

# 0 for threads
# 1 for operator
method = 1

function main()
    if length(ARGS) != 2
        println("Usage: " * PROGRAM_FILE * "  Inf/For/Bac")
        exit(-1)
    end

    adj = copy(read_vec("../adj.in", "Int")') .+ 1
    # adj = copy(readdlm(open("../adj.in"), Int)') .+ 1
    n_faces = size(adj)[2]
    in_feats = 13
    out_feats = 64
    x = copy(read_vec("../x.in", "Float32")')
    # x = copy(readdlm(open("../x.in"), Float32)')      # (n_faces, in_feats) -> (in_feats, n_faces)
    w0 = read_vec("../w0.in", "Float32")
    w1 = read_vec("../w1.in", "Float32")
    w2 = read_vec("../w2.in", "Float32")
    w3 = read_vec("../w3.in", "Float32")
    # w0 = readdlm(open("../w0.in"), Float32)     # (in_feats, out_feats)
    # w1 = readdlm(open("../w1.in"), Float32)
    # w2 = readdlm(open("../w2.in"), Float32)
    # w3 = readdlm(open("../w3.in"), Float32)
    y = zeros(Float32, (out_feats, n_faces))
    d_y = copy(read_vec("../d_y.in", "Float32")')
    # d_y = copy(readdlm(open("../d_y.in"), Float32)')
    if size(adj) != (3, n_faces)
        println("adj error")
    elseif size(x) != (in_feats, n_faces)
        println("x error")
    elseif size(w0) != (in_feats, out_feats) || size(w1) != (in_feats, out_feats)
        println("w error")
    end

    lambda = singl_conv
    if method == 1
        lambda = operator_conv
        adj = copy(adj')
        w0 = copy(w0')
        w1 = copy(w1')
        w2 = copy(w2')
        w3 = copy(w3')
    end
    if ARGS[2] == "Inf"
        warmup_num = 10
        test_num = 1000
        for i = 1:warmup_num
            y = lambda(adj, x, w0, w1, w2, w3, n_faces, in_feats, out_feats)
        end
        time = @timed begin
            for i = 1:test_num
                y = lambda(adj, x, w0, w1, w2, w3, n_faces, in_feats, out_feats)
            end
        end
        write_vec("y.out", Array(y))
        # writedlm("y.out", [@sprintf("%.18e", i) for i in Array(y')], ' ')
        println("Inference Time = " * string(time.time / test_num * 1000) * " ms")
        exit(0)
    elseif ARGS[2] == "For"
        warmup_num = 10
        test_num = 1000
        for i = 1:warmup_num
            z, back = Zygote.pullback(
                (x, w0, w1, w2, w3) -> sum(lambda(adj, x, w0, w1, w2, w3, n_faces, in_feats, out_feats) .* d_y),
                x, w0, w1, w2, w3
            )
            println("warmup: [" * string(i) * "/" * string(warmup_num) * "]  Done.")
        end
        time = @timed begin
            for i = 1:test_num
                z, back = Zygote.pullback(
                    (x, w0, w1, w2, w3) -> sum(lambda(adj, x, w0, w1, w2, w3, n_faces, in_feats, out_feats) .* d_y),
                    x, w0, w1, w2, w3
                )
                println("test: [" * string(i) * "/" * string(test_num) * "]  Done.")
            end
        end
        println("Forward Time = " * string(time.time / test_num * 1000) * " ms")
    elseif ARGS[2] == "Bac"
        warmup_num = 10
        test_num = 1000

        z, back = Zygote.pullback(
            (x, w0, w1, w2, w3) -> sum(lambda(adj, x, w0, w1, w2, w3, n_faces, in_feats, out_feats) .* d_y),
            x, w0, w1, w2, w3
        )

        for i = 1:warmup_num
            back_array = back(1)
            if i == 1
                write_vec("d_x.out", back_array[1])
                write_vec("d_w0.out", back_array[2])
                write_vec("d_w1.out", back_array[3])
                write_vec("d_w2.out", back_array[4])
                write_vec("d_w3.out", back_array[5])
            #    writedlm("d_x.out", [@sprintf("%.18e", i) for i in Array(back_array[1]')], ' ')
            #    writedlm("d_w0.out", [@sprintf("%.18e", i) for i in Array(back_array[2]')], ' ')
            #    writedlm("d_w1.out", [@sprintf("%.18e", i) for i in Array(back_array[3]')], ' ')
            #    writedlm("d_w2.out", [@sprintf("%.18e", i) for i in Array(back_array[4]')], ' ')
            #    writedlm("d_w3.out", [@sprintf("%.18e", i) for i in Array(back_array[5]')], ' ')
            end
            println("warmup: [" * string(i) * "/" * string(warmup_num) * "]  Done.")
        end
        time = @timed begin
            for i = 1:test_num
                back_array = back(1)
                println("test: [" * string(i) * "/" * string(test_num) * "]  Done.")
            end
        end
        println("Backward Time = " * string(time.time / test_num * 1000) * " ms")
    else
        println("Usage: " * PROGRAM_FILE * "  Inf/For/Bac")
        exit(-1)
    end
end

main()