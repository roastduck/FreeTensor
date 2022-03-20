using CUDA, Flux, Zygote
using IterTools

include("../../common/julia/io.jl")
include("../../common/julia/gpu.jl")

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

    vertices = copy(read_vec("../vertices.in", "Float32")')
    faces = copy(read_vec("../faces.in", "Int")') .+ 1
    n_verts = size(vertices)[2]
    n_faces = size(faces)[2]
    h = 64
    w = 64
    pixel1 = getindex.(product(range(0, 1, length=w), range(0, 1, length=h)), 2)
    pixel2 = getindex.(product(range(0, 1, length=w), range(0, 1, length=h)), 1)
    pixels = CuArray(cat(reshape(pixel1, (1, w, h)), reshape(pixel2, (1, w, h)), dims=1))
    y = zeros(Float32, (w, h, n_faces))
    d_y = read_vec("../d_y.in", "Float32")

    vertices = CuArray(vertices)
    faces = CuArray(faces)
    y = CuArray(y)
    pixels = CuArray(pixels)

    if ARGS[2] == "Inf"
        for i = 1:warmup_num
            y = rasterize(vertices, faces, pixels, h, w, n_verts, n_faces)
            if i == 1
                write_vec("y.out", Array(y))
            end
        end
        if haskey(ENV, "PROFILE_GPU")
            profile_start()
        end
        time = @timed begin
            for i = 1:test_num
                y = rasterize(vertices, faces, pixels, h, w, n_verts, n_faces)
            end
        end
        if haskey(ENV, "PROFILE_GPU")
            profile_stop()
        end
        println("Inference Time = " * string(time.time / test_num * 1000) * " ms")
    elseif ARGS[2] == "For"
        for i = 1:warmup_num
            z, back = Zygote.pullback(
                (vertices) -> sum(rasterize(vertices, faces, pixels, h, w, n_verts, n_faces) .* d_y),
                vertices
            )
            if i % 20 == 0
                println("warmup: [" * string(i) * "/" * string(warmup_num) * "]  Done.")
            end
        end
        time = @timed begin
            for i = 1:test_num
                z, back = Zygote.pullback(
                    (vertices) -> sum(rasterize(vertices, faces, pixels, h, w, n_verts, n_faces) .* d_y),
                    vertices
                )
                if i % 20 == 0
                    println("test: [" * string(i) * "/" * string(test_num) * "]  Done.")
                end
            end
        end
        println("Forward Time = " * string(time.time / test_num * 1000) * " ms")
    elseif ARGS[2] == "Bac"
        z, back = Zygote.pullback(
            (vertices) -> sum(rasterize(vertices, faces, pixels, h, w, n_verts, n_faces) .* d_y),
            vertices
        )
        for i = 1:warmup_num
            back_array = back(1)
            if i == 1
                write_vec("d_vertices.out", Array(back_array[1]))
            end
            if i % 20 == 0
                println("warmup: [" * string(i) * "/" * string(warmup_num) * "]  Done.")
            end
        end
        time = @timed begin
            for i = 1:test_num
                back_array = back(1)
                if i % 20 == 0
                    println("test: [" * string(i) * "/" * string(test_num) * "]  Done.")
                end
            end
        end
        println("Backward Time = " * string(time.time / test_num * 1000) * " ms")
    else
        println("Usage: " * PROGRAM_FILE * "  Inf/For/Bac")
        exit(-1)
    end
end

main()
