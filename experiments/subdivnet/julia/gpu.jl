using CUDA;
using Printf, DelimitedFiles;

function conv!(adj, x, w0, w1, w2, w3, n_faces, in_feats, out_feats)::CuArray{Float32}
    # adj_feat = reshape(x[:, Vector(reshape(adj, :))], (in_feats, 3, n_faces))
    adj_feat = reshape(x[:, Vector(reshape(adj, :))], (in_feats, n_faces, 3))   # (in_feats, n_faces, 3)

    # y0
    y = w0 * x

    # y1
    # y += w1 * dropdims(sum(adj_feat, dims=2), dims=2)
    y += w1 * dropdims(sum(adj_feat, dims=3), dims=3)

    # y2
    # y += w2 * dropdims(sum(abs.(adj_feat - cat(adj_feat[:, 2:3, :], reshape(adj_feat[:, 1, :], (in_feats, 1, n_faces)), dims=2)), dims=2), dims=2)
    y += w2 * dropdims(sum(abs.(adj_feat - cat(adj_feat[:, :, 2:3], reshape(adj_feat[:, :, 1], (in_feats, n_faces, 1)), dims=3)), dims=3), dims=3)

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

    adj = CuArray(adj)
    x = CuArray(x)
    w0 = CuArray(w0)
    w1 = CuArray(w1)
    w2 = CuArray(w2)
    w3 = CuArray(w3)
    y = CuArray(y)

    warmup_num = 10
    test_num = 1000

    for i = 1:warmup_num
        y = conv!(adj, x, w0, w1, w2, w3, n_faces, in_feats, out_feats)
    end
    # exit()
    time = @timed begin
        for i = 1:test_num
            y = conv!(adj, x, w0, w1, w2, w3, n_faces, in_feats, out_feats)
        end
    end
    writedlm("y.out", [@sprintf("%.18e", i) for i in Array(y')], ' ')
    println("Time = " * string(time.time / test_num * 1000) * " ms")
end

main()