using DelimitedFiles, Printf

"""
    Load a 3D object and returns the adjacency array of the faces


    Parameters
    ----------
    path: str
        Path to a 3D object file, where a `v <x> <y> <z>` line means there is a vertex at coordinate (x, y, z),
        a `f <i> <j> <k>` line means there is a face among vertices i, j and k. Faces are stored in conter-clockwise
        order


    Returns
    -------
    (np.array, np.array)
        ret[0] is an n*3-shaped numpy array, where n is the number of vertices. array[i] = the coordinate (x, y, z)
        ret[1] is an m*3-shaped numpy array, where m is the number of faces. array[i] = each vertices of the face
"""
function load_face(file)
    words = readlines(file)
    faces_size = 0
    vertices_size = 0
    for line in words
        if line[1] == 'f'
            faces_size += 1
        elseif line[1] == 'v'
            vertices_size += 1
        end
    end
    faces = zeros(Int, (3, faces_size))
    vertices = zeros(Float32, (3, vertices_size))
    faces_size, vertices_size = 0, 0
    for line in words
        if line[1] == 'f'
            faces_size += 1
            faces[:, faces_size] = map(x -> parse(Int, x), split(line, ' ')[2:end])
        elseif line[1] == 'v'
            vertices_size += 1
            vertices[:, vertices_size] = map(x -> parse(Float32, x), split(line, ' ')[2:end])
        end
    end
    return vertices, faces
end

function cross_product(v1::Tuple{Float32,Float32}, v2::Tuple{Float32,Float32})::Float32
    return v1[1] * v2[2] - v1[2] * v2[1]
end

function dot_product(v1::Tuple{Float32,Float32}, v2::Tuple{Float32,Float32})::Float32
    return v1[1] * v2[1] + v1[2] * v2[2]
end

using LinearAlgebra
using Flux
const sigma = 1e-4

function rasterize(vertices, faces, y, h, w, n_verts, n_faces)
    Threads.@threads for i = 1:n_faces
        # v = Vector{Tuple{Float32, Float32}}(undef, 3)
        # for p = 1:3
        #     v[p] = (vertices[1, faces[p, i]], vertices[2, faces[p, i]])
        # end
        v1 = (vertices[1, faces[1, i]], vertices[2, faces[1, i]])
        v2 = (vertices[1, faces[2, i]], vertices[2, faces[2, i]])
        v3 = (vertices[1, faces[3, i]], vertices[2, faces[3, i]])

        for j = 0:h-1, k=0:w-1
            pixel = (one(Float32) / (h-1) * j, one(Float32) / (w-1) * k)
            cp1 = cross_product(pixel .- v1, v2 .- v1)
            cp2 = cross_product(pixel .- v2, v3 .- v2)
            cp3 = cross_product(pixel .- v3, v1 .- v3)
            dist1 = dot_product(pixel .- v1, v2 .- v1) >= 0 ? (
                dot_product(pixel .- v2, v1 .- v2) >= 0 ?
                    abs(cp1) / norm(v2 .- v1) : norm(pixel .- v2)
                ) : norm(pixel .- v1)
            dist2 = dot_product(pixel .- v2, v3 .- v2) >= 0 ? (
                dot_product(pixel .- v3, v2 .- v3) >= 0 ?
                    abs(cp2) / norm(v3 .- v2) : norm(pixel .- v3)
                ) : norm(pixel .- v2)
            dist3 = dot_product(pixel .- v3, v1 .- v3) >= 0 ? (
                dot_product(pixel .- v1, v3 .- v1) >= 0 ?
                    abs(cp3) / norm(v1 .- v3) : norm(pixel .- v1)
                ) : norm(pixel .- v3)
            coeff = (cp1 < 0 && cp2 < 0 && cp3 < 0) ? 1 : -1
            dist = min(dist1, dist2, dist3)
            y[k+1, j+1, i] = sigmoid(coeff * dist * dist / sigma)
        end
    end
end

if length(ARGS) != 0
    println("Usage: " * PROGRAM_FILE)
    exit(-1)
end

vertices = copy(readdlm(open("../vertices.in"), Float32)')
faces = copy(readdlm(open("../faces.in"), Int)') .+ 1
const n_verts = size(vertices)[2]
const n_faces = size(faces)[2]
const h = 64
const w = 64
y = zeros(Float32, (w, h, n_faces))

warmup_num = 10
test_num = 100
for i = 1:warmup_num
    rasterize(vertices, faces, y, h, w, n_verts, n_faces)
    # writedlm("y.out", [@sprintf("%.10f", i) for i in reshape(y, (1, :))], '\n')
    # exit(0)
end
time = @timed begin
    for i = 1:test_num
        rasterize(vertices, faces, y, h, w, n_verts, n_faces)
    end
end
writedlm("y.out", [@sprintf("%.10f", i) for i in reshape(y, (1, :))], '\n')
println("Time = " * string(time.time / test_num * 1000) * " ms")
