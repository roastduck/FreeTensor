using CUDA
using Printf, DelimitedFiles
using Flux, Zygote

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
    if length(ARGS) != 1
        println("Usage: " * PROGRAM_FILE)
        exit(-1)
    end

    adj = readdlm(open("../adj.in"), Int) .+ 1   # (n_faces, 3)
    n_faces = size(adj)[1]
    in_feats = 13
    out_feats = 64
    x = copy(readdlm(open("../x.in"), Float32)')        # (n_faces, in_feats) -> (in_feats, n_faces)
    w0 = copy(readdlm(open("../w0.in"), Float32)')      # (in_feats, out_feats)
    w1 = copy(readdlm(open("../w1.in"), Float32)')
    w2 = copy(readdlm(open("../w2.in"), Float32)')
    w3 = copy(readdlm(open("../w3.in"), Float32)')
    y = zeros(Float32, (out_feats, n_faces))
    d_y = copy(readdlm(open("../d_y.in"), Float32)')

    adj = CuArray(adj)
    x = CuArray(x)
    w0 = CuArray(w0)
    w1 = CuArray(w1)
    w2 = CuArray(w2)
    w3 = CuArray(w3)
    y = CuArray(y)
    d_y = CuArray(d_y)

    warmup_num = 1
    test_num = 1

    for i = 1:warmup_num
        y = my_conv(adj, x, w0, w1, w2, w3, n_faces, in_feats, out_feats)
    end
    # exit()
    time = @timed begin
        for i = 1:test_num
            y = my_conv(adj, x, w0, w1, w2, w3, n_faces, in_feats, out_feats)
        end
    end
    writedlm("y.out", [@sprintf("%.18e", i) for i in Array(y')], ' ')
    println("Inference Time = " * string(time.time / test_num * 1000) * " ms")
exit(0)

    for i = 1:warmup_num
        # gs = gradient(params(x, w0, w1, w2, w3)) do
        #     z = sum(my_conv(adj, x, w0, w1, w2, w3, n_faces, in_feats, out_feats) .* d_y)
        # end
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


    z, back = Zygote.pullback(
        (x, w0, w1, w2, w3) -> sum(my_conv(adj, x, w0, w1, w2, w3, n_faces, in_feats, out_feats) .* d_y),
        x, w0, w1, w2, w3
    )
    for i = 1:warmup_num
        # gs = gradient(params(x, w0, w1, w2, w3)) do
        #     z = sum(my_conv(adj, x, w0, w1, w2, w3, n_faces, in_feats, out_feats) .* d_y)
        # end
        back_array = back(1)
        if i == 1
            writedlm("d_x.out", [@sprintf("%.18e", i) for i in Array(back_array[1]')], ' ')
            writedlm("d_w0.out", [@sprintf("%.18e", i) for i in Array(back_array[2]')], ' ')
            writedlm("d_w1.out", [@sprintf("%.18e", i) for i in Array(back_array[3]')], ' ')
            writedlm("d_w2.out", [@sprintf("%.18e", i) for i in Array(back_array[4]')], ' ')
            writedlm("d_w3.out", [@sprintf("%.18e", i) for i in Array(back_array[5]')], ' ')
        end
    end
    # exit()
    time = @timed begin
        for i = 1:test_num
            back_array = back(1)
        end
    end
    println("Backward Time = " * string(time.time / test_num * 1000) * " ms")
end

main()