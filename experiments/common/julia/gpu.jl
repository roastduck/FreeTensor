function profile_start()
    ccall((:cudaProfilerStart, "libcudart"), Cvoid, ())
end

function profile_stop()
    ccall((:cudaProfilerStop, "libcudart"), Cvoid, ())
end
