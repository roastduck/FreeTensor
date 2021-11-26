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

function cross_product(v1::Tuple{Float32,Float32}, v2::Tuple{Float32,Float32})
    return v1[1] * v2[2] - v1[2] * v2[1]
end

using LinearAlgebra
const sigma = 1e-4

function rasterize(vertices::Matrix{Float32}, faces::Matrix{Int}, y::Array{Float32, 3}, h::Int, w::Int, n_verts::Int, n_faces::Int)
    Threads.@threads for i = 1:n_faces
        v1 = (vertices[1, faces[1, i]], vertices[2, faces[1, i]])
        v2 = (vertices[1, faces[2, i]], vertices[2, faces[2, i]])
        v3 = (vertices[1, faces[3, i]], vertices[2, faces[3, i]])
        e1 = v2 .- v1
        e2 = v3 .- v2
        e3 = v1 .- v3
        len1 = norm(e1)::Float32
        len2 = norm(e2)::Float32
        len3 = norm(e3)::Float32
        for j = 1:h, k = 1:w
            pixel = (one(Float32) / (h - 1) * j, one(Float32) / (w - 1) * k)

            p1 = pixel .- v1
            p2 = pixel .- v2
            p3 = pixel .- v3
            cp1 = cross_product(p1, e1)::Float32
            cp2 = cross_product(p2, e2)::Float32
            cp3 = cross_product(p3, e3)::Float32
            dist1 = norm(p1)::Float32
            dist2 = norm(p2)::Float32
            dist3 = norm(p3)::Float32

            dist = min(min(abs(cp1) / len1, abs(cp2) / len2, abs(cp3) / len3), min(dist1, dist2, dist3))

            if cp1 < 0 && cp2 < 0 && cp3 < 0
                coeff = 1
            else
                coeff = -1
            end
            y[k, j, i] = coeff * dist * dist / sigma
        end
    end
end

if length(ARGS) != 1
    println("Usage: " * PROGRAM_FILE * " <obj-file>")
    exit(-1)
end
const obj_file = ARGS[1]

const vertices, faces = load_face(obj_file)
const n_verts = size(vertices)[2]
const n_faces = size(faces)[2]
const h = 64
const w = 64
y = zeros(Float32, (w, h, n_faces))

test_num = 100
rasterize(vertices, faces, y, h, w, n_verts, n_faces)
time = @timed begin
    for i = 1:test_num
        rasterize(vertices, faces, y, h, w, n_verts, n_faces)
    end
end
println("Time = " * string(time.time / test_num * 1000) * " ms")
