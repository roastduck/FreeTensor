using CUDA
using Flux, Zygote

include("../../common/julia/io.jl")

function my_conv(adj, x, w0, w1, w2, w3, n_faces, in_feats, out_feats)::CuArray{Float32}
    # adj_feat = reshape(x[:, Vector(reshape(adj, :))], (in_feats, 3, n_faces))
    adj_feat = reshape(x[:, reshape(adj, :)], (in_feats, n_faces, 3))   # (in_feats, n_faces, 3)

    # y0
    y = w0 * x

    # y1
    # y += w1 * dropdims(sum(adj_feat, dims=2), dims=2)
    y += w1 * dropdims(sum(adj_feat, dims=3), dims=3)

    # y2
    # y += w2 * dropdims(sum(abs.(adj_feat - cat(adj_feat[:, 2:3, :], reshape(adj_feat[:, 1, :], (in_feats, 1, n_faces)), dims=2)), dims=2), dims=2)
    y += w2 * dropdims(sum(abs.(adj_feat .- cat(adj_feat[:, :, 2:3], reshape(adj_feat[:, :, 1], (in_feats, n_faces, 1)), dims=3)), dims=3), dims=3)

    # y3
    # y += w3 * dropdims(sum(abs.(broadcast(-, adj_feat, reshape(x, (in_feats, 1, n_faces)))), dims=2), dims=2)
    y += w3 * dropdims(sum(abs.(broadcast(-, adj_feat, reshape(x, (in_feats, n_faces, 1)))), dims=3), dims=3)
    return y
end

function main()
    warmup_num = 10
    test_num = 100
    if length(ARGS) != 2 && length(ARGS) != 4
        println("Usage: " * PROGRAM_FILE * "  Inf/For/Bac  <warmup_repeat> <timing_repeat>")
        exit(-1)
    end
    if length(ARGS) == 4
        warmup_num = parse(Int, ARGS[3])
        test_num = parse(Int, ARGS[4])
    end
    println(warmup_num, " warmup, ", test_num, "repeats for evalution")

    adj = copy(read_vec("../adj.in", "Int")) .+ 1
    n_faces = size(adj)[1]
    in_feats = 13
    out_feats = 64
    x = copy(read_vec("../x.in", "Float32")')
    w0 = copy(read_vec("../w0.in", "Float32")')
    w1 = copy(read_vec("../w1.in", "Float32")')
    w2 = copy(read_vec("../w2.in", "Float32")')
    w3 = copy(read_vec("../w3.in", "Float32")')
    y = zeros(Float32, (out_feats, n_faces))
    d_y = copy(read_vec("../d_y.in", "Float32")')

    adj = CuArray(adj)
    x = CuArray(x)
    w0 = CuArray(w0)
    w1 = CuArray(w1)
    w2 = CuArray(w2)
    w3 = CuArray(w3)
    y = CuArray(y)
    d_y = CuArray(d_y)

    if ARGS[2] == "Inf"
        for i = 1:warmup_num
            y = my_conv(adj, x, w0, w1, w2, w3, n_faces, in_feats, out_feats)
        end
        # exit()
        time = @timed begin
            for i = 1:test_num
                y = my_conv(adj, x, w0, w1, w2, w3, n_faces, in_feats, out_feats)
            end
        end
        write_vec("y.out", Array(y))
        println("Inference Time = " * string(time.time / test_num * 1000) * " ms")
    
    elseif ARGS[2] == "For"
        for i = 1:warmup_num
            z, back = Zygote.pullback(
                (x, w0, w1, w2, w3) -> sum(my_conv(adj, x, w0, w1, w2, w3, n_faces, in_feats, out_feats) .* d_y),
                x, w0, w1, w2, w3
            )
        end
        # exit()
        time = @timed begin
            for i = 1:test_num
                z, back = Zygote.pullback(
                    (x, w0, w1, w2, w3) -> sum(my_conv(adj, x, w0, w1, w2, w3, n_faces, in_feats, out_feats) .* d_y),
                    x, w0, w1, w2, w3
                )
            end
        end
        println("Forward Time = " * string(time.time / test_num * 1000) * " ms")

    elseif ARGS[2] == "Bac"
        z, back = Zygote.pullback(
            (x, w0, w1, w2, w3) -> sum(my_conv(adj, x, w0, w1, w2, w3, n_faces, in_feats, out_feats) .* d_y),
            x, w0, w1, w2, w3
        )
        for i = 1:warmup_num
            back_array = back(1)
            if i == 1
                write_vec("d_x.out", Array(back_array[1]))
                write_vec("d_w0.out", Array(back_array[2]))
                write_vec("d_w1.out", Array(back_array[3]))
                write_vec("d_w2.out", Array(back_array[4]))
                write_vec("d_w3.out", Array(back_array[5]))
            end
        end
        time = @timed begin
            for i = 1:test_num
                back_array = back(1)
            end
        end
        println("Backward Time = " * string(time.time / test_num * 1000) * " ms")
    else
        println("Usage: " * PROGRAM_FILE * "  Inf/For/Bac")
        exit(-1)
    end
end

main()