using DelimitedFiles, Printf
using Zygote, Flux
using IterTools

function cross_product(v1::Tuple{Float32,Float32}, v2::Tuple{Float32,Float32})::Float32
    return v1[1] * v2[2] - v1[2] * v2[1]
end

function dot_product(v1::Tuple{Float32,Float32}, v2::Tuple{Float32,Float32})::Float32
    return v1[1] * v2[1] + v1[2] * v2[2]
end

using LinearAlgebra
using Flux
const sigma = 1e-4

function para_rasterize(vertices, faces, h, w, n_verts, n_faces)
    y = zeros(Float32, (w, h, n_faces))
    Threads.@threads for i = 1:n_faces
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
    return y
end

function rasterize(vertices, faces, pixels, h, w, n_verts, n_faces)
    sigma = 1e-4

    # pixels = CuArray{Int}(undef, (2, w, h))

    face_verts = reshape(vertices[:, reshape(faces, (:))], (3, 3, n_faces))[1:2, :, :]

    norm(v) = sqrt.(selectdim(v, 1, 1) .^ 2 .+ selectdim(v, 1, 2) .^ 2)
    cross_product(v1, v2) = selectdim(v1, 1, 1) .* selectdim(v2, 1, 2) .- selectdim(v1, 1, 2) .* selectdim(v2, 1, 1)
    dot_product(v1, v2) = selectdim(v1, 1, 1) .* selectdim(v2, 1, 1) .+ selectdim(v1, 1, 2) .* selectdim(v2, 1, 2)
    vert_clockwise(v1, v2, pixel) = (cross_product(pixel .- v1, v2 .- v1) .< 0)
    inside_face(v1, v2, v3, pixel) = vert_clockwise(v1, v2, pixel) .& vert_clockwise(v2, v3, pixel) .& vert_clockwise(v3, v1, pixel)
    is_inside = inside_face(reshape(face_verts[:, 1, :], (2, 1, 1, n_faces)), reshape(face_verts[:, 2, :], (2, 1, 1, n_faces)), reshape(face_verts[:, 3, :], (2, 1, 1, n_faces)), reshape(pixels, (2, w, h, 1)))

    ternary(cond, val1, val2) = cond ? val1 : val2

    dist_pixel_to_seg(v1, v2, pixel) = ternary.(dot_product(pixel .- v1, v2 .- v1) .>= 0,
        ternary.(dot_product(pixel .- v2, v1 .- v2) .>= 0,
            abs.(cross_product(pixel .- v1, v2 .- v1)) ./ norm(v2 .- v1),
            norm(pixel .- v2)
        ), norm(pixel .- v1))
    dist_pixel_to_face(v1, v2, v3, pixel) = min.(
        dist_pixel_to_seg(v1, v2, pixel),
        dist_pixel_to_seg(v2, v3, pixel),
        dist_pixel_to_seg(v3, v1, pixel)
    )
    dist = dist_pixel_to_face(reshape(face_verts[:, 1, :], (2, 1, 1, n_faces)), reshape(face_verts[:, 2, :], (2, 1, 1, n_faces)), reshape(face_verts[:, 3, :], (2, 1, 1, n_faces)), reshape(pixels, (2, w, h, 1)))

    d = ternary.(is_inside, 1, -1) .* (dist .^ 2) ./ sigma
    d = sigmoid.(d)
    return d
end

function main()
    if length(ARGS) != 2
        println("Usage: " * PROGRAM_FILE * " Inf/For/Bac")
        exit(-1)
    end

    vertices = copy(readdlm(open("../vertices.in"), Float32)')
    faces = copy(readdlm(open("../faces.in"), Int)') .+ 1
    n_verts = size(vertices)[2]
    n_faces = size(faces)[2]
    h = 64
    w = 64
    pixel1 = getindex.(product(range(0, 1, length=w), range(0, 1, length=h)), 1)
    pixel2 = getindex.(product(range(0, 1, length=w), range(0, 1, length=h)), 2)
    pixels = cat(reshape(pixel1, (1, w, h)), reshape(pixel2, (1, w, h)), dims=1)
    d_y = reshape(readdlm(open("../d_y.in"), Float32), (w, h, n_faces))

    if ARGS[2] == "Inf"
        warmup_num = 10
        test_num = 100
        for i = 1:warmup_num
            y = para_rasterize(vertices, faces, h, w, n_verts, n_faces)
            if i == 1
                writedlm("y.out", [@sprintf("%.10f", i) for i in reshape(y, (1, :))], '\n')
            end
            println("warmup: [" * string(i) * "/" * string(warmup_num) * "]  Done.")
        end
        time = @timed begin
            for i = 1:test_num
                y = para_rasterize(vertices, faces, h, w, n_verts, n_faces)
                println("test: [" * string(i) * "/" * string(test_num) * "]  Done.")
            end
        end
        println("Inference Time = " * string(time.time / test_num * 1000) * " ms")
    elseif ARGS[2] == "For"
        warmup_num = 2
        test_num = 5
        for i = 1:warmup_num
            z, back = Zygote.pullback(
                (vertices) -> sum(rasterize(vertices, faces, pixels, h, w, n_verts, n_faces) .* d_y),
                vertices
            )
            println("warmup: [" * string(i) * "/" * string(warmup_num) * "]  Done.")
            # exit(0)
        end
        time = @timed begin
            for i = 1:test_num
                z, back = Zygote.pullback(
                    (vertices) -> sum(rasterize(vertices, faces, pixels, h, w, n_verts, n_faces) .* d_y),
                    vertices
                )
                println("test: [" * string(i) * "/" * string(test_num) * "]  Done.")
            end
        end
        println("Forward Time = " * string(time.time / test_num * 1000) * " ms")
    elseif ARGS[2] == "Bac"
        warmup_num = 2
        test_num = 10
        z, back = Zygote.pullback(
            (vertices) -> sum(rasterize(vertices, faces, pixels, h, w, n_verts, n_faces) .* d_y),
            vertices
        )
        for i = 1:warmup_num
            back_array = back(1)
            if i == 1
                writedlm("d_vertices.out", [@sprintf("%.18e", i) for i in reshape(Array(back_array[1]), :)], ' ')
            end
            println("warmup: [" * string(i) * "/" * string(warmup_num) * "]  Done.")
        end
        time = @timed begin
            for i = 1:test_num
                back(1)
                println("test: [" * string(i) * "/" * string(test_num) * "]  Done.")
            end
        end
        println("Backward Time = " * string(time.time / test_num * 1000) * " ms")
    else
        println("Usage: " * PROGRAM_FILE * " Inf/For/Bac")
        exit(-1)
    end
end

main()