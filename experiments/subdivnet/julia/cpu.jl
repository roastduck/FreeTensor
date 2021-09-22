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

    ret = []
    for i = 1:length(faces)
        append!(ret, [(edgeToFaces[faces[i][2], faces[i][1]], edgeToFaces[faces[i][3], faces[i][2]], edgeToFaces[faces[i][1], faces[i][3]])])
    end

    return ret
end

function conv(adj, x, w0, w1, w2, w3, y, n_faces, in_feats, out_feats)
    Threads.@threads for i = 1:n_faces
        sum1 = zeros(Float32, in_feats)
        sum2 = zeros(Float32, in_feats)
        sum3 = zeros(Float32, in_feats)
        for k = 1:in_feats
            for p = 1:3
                sum1[k] += x[adj[i][p], k]
                sum2[k] += abs(x[adj[i][p], k] - x[adj[i][p % 3 + 1], k])
                sum3[k] += abs(x[adj[i][p], k] - x[i, k])
            end
        end
        for j = 1:out_feats
            y[i, j] = 0.
            for k = 1:in_feats
                y[i, j] += x[i, k] * w0[k, j] + sum1[k] * w1[k, j] + sum2[k] * w2[k, j] + sum3[k] * w3[k, j]
            end
        end
    end
end

if length(ARGS) != 1
    println("Usage: " * PROGRAM_FILE * " <obj-file>")
    exit(-1)
end
obj_file = open(ARGS[1])

adj = load_face(obj_file)
n_faces = size(adj)[1]
in_feats = 13
out_feats = 64
x = rand(Float32, (n_faces, in_feats))
w0 = rand(Float32, (in_feats, out_feats))
w1 = rand(Float32, (in_feats, out_feats))
w2 = rand(Float32, (in_feats, out_feats))
w3 = rand(Float32, (in_feats, out_feats))
y = zeros(Float32, (n_faces, out_feats))

test_num = 1000
conv(adj, x, w0, w1, w2, w3, y, n_faces, in_feats, out_feats)
time = @timed begin
    for i = 1:test_num
        conv(adj, x, w0, w1, w2, w3, y, n_faces, in_feats, out_feats)
    end
end
println("Time = " * string(time.time / test_num * 1000) * " ms")
