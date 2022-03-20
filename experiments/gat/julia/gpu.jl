using Printf
using Flux, GraphNeuralNetworks
using CUDA
using Flux: cpu, gpu

include("../../common/julia/io.jl")
include("../../common/julia/gpu.jl")

device = CUDA.functional() ? gpu : cpu
println("Running on ", CUDA.functional() ? "gpu" : "cpu")

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

_ptr = read_vec("../ptr.in", "Int")
ptr = reshape(_ptr .+ 1, :)  # (num_v + 1)
_idx = read_vec("../idx.in", "Int")
idx = reshape(_idx .+ 1, :)  # (num_e)
num_v = length(ptr) - 1
num_e = length(idx)

feat_len = 32
X = copy(read_vec("../x.in", "Float32")') |> device
# X = copy(readdlm(open("../x.in"), Float32)') |> device  # (feat_len, num_v)
# w = copy(readdlm(open("../w.in"), Float32)')   # (feat_len, feat_len)
# w_attn_1::Vector{Float32} = reshape(readdlm(open("../w_attn_1.in"), Float32), :) # (feat_len)
# w_attn_2::Vector{Float32} = reshape(readdlm(open("../w_attn_2.in"), Float32), :) # (feat_len)
# y::Matrix{Float32} = zeros(Float32, (feat_len, num_v))   # (feat_len, num_v)

feat_len_in, feat_len_out = 32, 32
# num_v = 4
# num_e = 4
# ptr=[1 2 3 4 4]
# idx=[2 3 4]
# CSR -> adjacency list
adj=Array{Int,1}[]
for i =1:num_v
    Z = Vector{Int}()
    for j = ptr[i]:ptr[i+1]-1
        push!(Z,idx[j])
    end
    push!(adj, Z)
end

g = GNNGraph(adj) |> device
# println("# vertices = ", g.num_nodes)
# println("# edges = ", g.num_edges)

model = GNNChain(GATConv(feat_len_in => feat_len_out)) |> device

y = model(g, X)
write_vec("y.out", Array(y))
for i = 1:warmup_num
    y = model(g, X)
end

if haskey(ENV, "PROFILE_GPU")
    profile_start()
end
time = @timed begin
    for i = 1:test_num
        y = model(g, X)
    end
end
if haskey(ENV, "PROFILE_GPU")
    profile_stop()
end
println("Inference Time = " * string(time.time / test_num * 1000) * " ms")
