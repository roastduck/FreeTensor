using Printf
using Zygote

include("../../common/julia/io.jl")

function inference(ptr::Vector{Int}, idx::Vector{Int},
    feat::Matrix{Float32}, weight::Matrix{Float32},
    attn_l::Vector{Float32}, attn_r::Vector{Float32}, y::Matrix{Float32},
     num_v, num_e, feat_len)

    @inbounds feat2::Matrix{Float32} = weight * feat
    @inbounds att_l::Vector{Float32} = reshape(attn_l' * feat2, :)
    @inbounds att_r::Vector{Float32} = reshape(attn_r' * feat2, :)

    edge = zeros(Float32, num_e)
    edge_exp = zeros(Float32, num_e)

    @inbounds Threads.@threads for i = 1 : num_v
        edge_max = -Inf
        for k = ptr[i] : ptr[i+1]-1
            e = att_l[idx[k]] + att_r[i]
            edge[k] = (e >= 0 ? e : e * 0.1)
            edge_max = max(edge_max, edge[k])
        end
        edge_sum = zero(Float32)
        for k = ptr[i] : ptr[i+1]-1
            edge_exp[k] = exp(edge[k] - edge_max)
            edge_sum += edge_exp[k];
        end
        for j = 1 : feat_len
            y[j, i] = 0
            for k = ptr[i] : ptr[i+1]-1
                y[j, i] += feat2[j, idx[k]] * edge_exp[k] / edge_sum
            end
        end
    end
    return nothing
end

function main()
    warmup_num = 10
    test_num = 100
    if length(ARGS) != 1 && length(ARGS) != 3
        println("Usage: " * PROGRAM_FILE * " cpu/gpu <warmup_repeat> <timing_repeat>")
        exit(-1)
    end
    if length(ARGS) == 3
        warmup_num = parse(Int, ARGS[2])
        test_num = parse(Int, ARGS[3])
    end
    println(warmup_num, " warmup, ", test_num, "repeats for evalution")


    # ptr::Vector{Int} = reshape(readdlm(open("../ptr.in"), Int) .+ 1, :)  # (num_v + 1)
    # idx::Vector{Int} = reshape(readdlm(open("../idx.in"), Int) .+ 1, :)  # (num_e)
    ptr::Vector{Int} = read_vec("../ptr.in", "Int") .+ 1
    idx::Vector{Int} = read_vec("../idx.in", "Int") .+ 1
    num_v::Int = length(ptr) - 1
    num_e::Int = length(idx)

    feat_len::Int = 32
    x::Matrix{Float32} = copy(read_vec("../x.in", "Float32")')
    w::Matrix{Float32} = copy(read_vec("../w.in", "Float32")')
    # x::Matrix{Float32} = copy(readdlm(open("../x.in"), Float32)')   # (feat_len, num_v)
    # w::Matrix{Float32} = copy(readdlm(open("../w.in"), Float32)')   # (feat_len, feat_len)
    w_attn_1::Vector{Float32} = read_vec("../w_attn_1.in", "Float32")
    w_attn_2::Vector{Float32} = read_vec("../w_attn_2.in", "Float32")
    # w_attn_1::Vector{Float32} = reshape(readdlm(open("../w_attn_1.in"), Float32), :) # (feat_len)
    # w_attn_2::Vector{Float32} = reshape(readdlm(open("../w_attn_2.in"), Float32), :) # (feat_len)
    y::Matrix{Float32} = zeros(Float32, (feat_len, num_v))   # (feat_len, num_v)

    for i = 1:warmup_num
        inference(ptr, idx, x, w, w_attn_1, w_attn_2, y, num_v, num_e, feat_len)
    end
    time = @timed begin
        for i = 1:test_num
            inference(ptr, idx, x, w, w_attn_1, w_attn_2, y, num_v, num_e, feat_len)
        end
    end
    write_vec("y.out", y)
    println("Inference Time = " * string(time.time / test_num * 1000) * " ms")
end

main()