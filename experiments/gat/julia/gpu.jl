using Printf, DelimitedFiles
using Flux, GraphNeuralNetworks
using CUDA
using Flux: cpu, gpu

device = CUDA.functional() ? gpu : cpu
println(CUDA.functional())

ptr_float = round.(Int, readdlm(open("../ptr.in"), Float32))  # (num_v + 1)
ptr = reshape(ptr_float .+ 1, :)  # (num_v + 1)
idx_float = round.(Int, readdlm(open("../idx.in"), Float32))  # (num_v + 1)
idx = reshape(idx_float .+ 1, :)  # (num_e)
num_v = length(ptr) - 1
num_e = length(idx)

feat_len = 32
X = copy(readdlm(open("../x.in"), Float32)') |> device  # (feat_len, num_v)
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
println("# vertices = ", g.num_nodes)
println("# edges = ", g.num_edges)

model = GNNChain(GATConv(feat_len_in => feat_len_out)) |> device

warmup_num = 10
test_num = 1000

y = model(g, X)
writedlm("y.out", [@sprintf("%.18e", i) for i in Array(y')], ' ')
for i = 1:warmup_num
    y = model(g, X)
end

time = @timed begin
    for i = 1:test_num
        y = model(g, X)
    end
end
println("Inference Time = " * string(time.time / test_num * 1000) * " ms")
