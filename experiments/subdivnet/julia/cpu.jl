"""
    Load a 3D object and returns the adjacency array of the faces


    Parameters
    ----------
    path: str
        Path to a 3D object file, where a `f <i> <j> <k>` line means there is a face among point i, j and k


    Returns
    -------
    np.array
        An n*3-shaped numpy array, where n is the number of faces. array[i][j] = ID of the j-th adjacent face of the i-th face
"""

using DelimitedFiles, Printf;

function load_face(file)
    n = 0
    faces = []
    for line in eachline(file)
        if line[1] == 'f'
            append!(faces, [map(x -> parse(Int, x), split(line, ' ')[2:end])])
        elseif line[1] == 'v'
            n += 1
        end
    end

    edgeToFaces = zeros(Int, (n, n))
    for i = 1:length(faces)
        edgeToFaces[faces[i][1], faces[i][2]] = i
        edgeToFaces[faces[i][2], faces[i][3]] = i
        edgeToFaces[faces[i][3], faces[i][1]] = i
    end

    ret = zeros(Int, (3, length(faces)))
    for i = 1:length(faces)
        ret[1, i] = edgeToFaces[faces[i][2], faces[i][1]]
        ret[2, i] = edgeToFaces[faces[i][3], faces[i][2]]
        ret[3, i] = edgeToFaces[faces[i][1], faces[i][3]]
    end

    return ret
end

function conv!(adj, x, w0, w1, w2, w3, y, n_faces, in_feats, out_feats)
    y .= 0
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
end

function main()
    if length(ARGS) != 1
        println("Usage: " * PROGRAM_FILE)
        exit(-1)
    end

    adj = copy(readdlm(open("../adj.in"), Int)') .+ 1
    n_faces = size(adj)[2]
    in_feats = 13
    out_feats = 64
    x = copy(readdlm(open("../x.in"), Float32)')      # (n_faces, in_feats) -> (in_feats, n_faces)
    w0 = readdlm(open("../w0.in"), Float32)     # (in_feats, out_feats)
    w1 = readdlm(open("../w1.in"), Float32)
    w2 = readdlm(open("../w2.in"), Float32)
    w3 = readdlm(open("../w3.in"), Float32)
    y = zeros(Float32, (out_feats, n_faces))
    if size(adj) != (3, n_faces)
        println("adj error")
    elseif size(x) != (in_feats, n_faces)
        println("x error")
    elseif size(w0) != (in_feats, out_feats) || size(w1) != (in_feats, out_feats)
        println("w error")
    end

    warmup_num = 10
    test_num = 1000

    for i = 1:warmup_num
        conv!(adj, x, w0, w1, w2, w3, y, n_faces, in_feats, out_feats)
    end
    time = @timed begin
        for i = 1:test_num
            conv!(adj, x, w0, w1, w2, w3, y, n_faces, in_feats, out_feats)
        end
    end
    writedlm("y.out", [@sprintf("%.18e", i) for i in Array(y')], ' ')
    println("Time = " * string(time.time / test_num * 1000) * " ms")
end

main()