using DelimitedFiles, Printf

using CUDA, Flux, Zygote
using IterTools

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
        println("Usage: " * PROGRAM_FILE * "  Inf/For/Bac")
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
    pixels = CuArray(cat(reshape(pixel1, (1, w, h)), reshape(pixel2, (1, w, h)), dims=1))
    y = zeros(Float32, (w, h, n_faces))
    d_y = reshape(readdlm(open("../d_y.in"), Float32), (w, h, n_faces))

    vertices = CuArray(vertices)
    faces = CuArray(faces)
    y = CuArray(y)
    pixels = CuArray(pixels)

    if ARGS[2] == "Inf"
        warmup_num = 10
        test_num = 100
        for i = 1:warmup_num
            y = rasterize(vertices, faces, pixels, h, w, n_verts, n_faces)
            println("warmup: [" * string(i) * "/" * string(warmup_num) * "]  Done.")
        end
        time = @timed begin
            for i = 1:test_num
                y = rasterize(vertices, faces, pixels, h, w, n_verts, n_faces)
                println("test: [" * string(i) * "/" * string(test_num) * "]  Done.")
            end
        end
        writedlm("y.out", [@sprintf("%.10f", i) for i in reshape(Array(y), (1, :))], '\n')
        println("Inference Time = " * string(time.time / test_num * 1000) * " ms")
    elseif ARGS[2] == "For"
        warmup_num = 10
        test_num = 10
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
        warmup_num = 10
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