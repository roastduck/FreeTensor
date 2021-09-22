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
    vertices = []
    faces = []
    for line in eachline(file)
        if line[1] == 'f'
            append!(faces, [map(x -> parse(Int, x), split(line, ' ')[2:end])])
        elseif line[1] == 'v'
            append!(vertices, [map(x -> parse(Float32, x), split(line, ' ')[2:end])])
        end
    end
    return vertices, faces
end

function cross_product(v1, v2)
    return v1[1] * v2[2] - v1[2] * v2[1]
end

function norm(v)
    return sqrt(v[1] * v[1] + v[2] * v[2])
end

function sub(v1, v2)
    return [v1[1] - v2[1], v1[2] - v2[2]]
end

sigma = 1e-4

function rasterize(vertices, faces, y, h, w, n_verts, n_faces)
    Threads.@threads for i = 1:n_faces
        v1 = [vertices[faces[i][1]][1], vertices[faces[i][1]][2]]
        v2 = [vertices[faces[i][2]][1], vertices[faces[i][2]][2]]
        v3 = [vertices[faces[i][3]][1], vertices[faces[i][3]][2]]
        e1 = sub(v2, v1)
        e2 = sub(v3, v2)
        e3 = sub(v1, v3)
        len1 = norm(e1)
        len2 = norm(e2)
        len3 = norm(e3)
        for j = 1:h
            for k = 1:w
                pixel = [1. / (h - 1) * j, 1. / (w - 1) * k]

                p1 = sub(pixel, v1)
                p2 = sub(pixel, v2)
                p3 = sub(pixel, v3)
                cp1 = cross_product(p1, e1)
                cp2 = cross_product(p2, e2)
                cp3 = cross_product(p3, e3)
                dist1 = norm(p1)
                dist2 = norm(p2)
                dist3 = norm(p3)

                dist = min(min(abs(cp1) / len1, abs(cp2) / len2, abs(cp3) / len3), min(dist1, dist2, dist3))

                if cp1 < 0 && cp2 < 0 && cp3 < 0
                    coeff = 1
                else
                    coeff = -1
                end
                y[i, j, k] = coeff * dist * dist / sigma
            end
        end
    end
end

if length(ARGS) != 1
    println("Usage: " * PROGRAM_FILE * " <obj-file>")
    exit(-1)
end
obj_file = open(ARGS[1])

vertices, faces = load_face(obj_file)
n_verts = size(vertices)[1]
n_faces = size(faces)[1]
h = 64
w = 64
y = zeros(Float32, (n_faces, h, w))

test_num = 1000
rasterize(vertices, faces, y, h, w, n_verts, n_faces)
time = @timed begin
    for i = 1:test_num
        rasterize(vertices, faces, y, h, w, n_verts, n_faces)
    end
end
println("Time = " * string(time.time / test_num * 1000) * " ms")
