using Printf
using Flux, GraphNeuralNetworks
using CUDA
using Flux: cpu, gpu

include("../../common/julia/io.jl")

device = CUDA.functional() ? gpu : cpu
println(CUDA.functional())

_ptr = read_vec("../ptr.in", "Int")
ptr = reshape(_ptr .+ 1, :)  # (num_v + 1)
_idx = read_vec("../idx.in", "Int")
idx = reshape(_idx .+ 1, :)  # (num_e)
num_v = length(ptr) - 1
num_e = length(idx)

feat_len_in, feat_len_out = 32, 32 
X = copy(read_vec("../x.in", "Float32")') |> device
w = copy(read_vec("../w.in", "Float32")')
w_attn_1 = read_vec("../w_attn_1.in", "Float32")
w_attn_2 = read_vec("../w_attn_2.in", "Float32")

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

#model = GNNChain(GATConv(feat_len_in => feat_len_out)) |> device

# We use the internal API in order to pass in our own weight matrices. This API is dedicated to GraphNeuralNetworks v0.3.4
w_attn = reshape(vcat(w_attn_1, w_attn_2), (2 * feat_len_out, 1))
model = GNNChain(GATConv(w, false, w_attn, identity, Float32(0.1), feat_len_in => feat_len_out, 1, true)) |> device

warmup_num = 10
test_num = 1000

y = model(g, X)
write_vec("y.out", Array(y))

#for i = 1:warmup_num
#    y = model(g, X)
#end
#
#time = @timed begin
#    for i = 1:test_num
#        y = model(g, X)
#    end
#end
#println("Inference Time = " * string(time.time / test_num * 1000) * " ms")
