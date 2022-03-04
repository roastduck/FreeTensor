using CUDA, Zygote

include("../../common/julia/io.jl")

gpu = true
# gpu = false

function my_conv(a, b)
    adj_feat = a[:, reshape(b, :)]
    return adj_feat
end

function main()
    a = Float32.(copy(reshape(1:10:60, (2, 3))))
    b = reshape([1, 2, 1, 2, 1, 1, 1, 2], (4, 2))
    d_y = Float32.(ones(2, 8))

    for i = 0:1
        if i == 1
            a = CuArray(a)
            b = CuArray(b)
            d_y = CuArray(d_y)
        end

        z, back = Zygote.pullback(
            (a, b) -> sum(my_conv(a, b) .* d_y),
            a, b
        )

        back_array = back(1)
        write_vec(i == 0 ? "cpu.out" : "gpu.out", Array(back_array[1]))
    end
end

main()