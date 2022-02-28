
function read_vec(file::String, type::String)
    ls = readlines(file)
    shape = tuple(parse.(Int, split(ls[1]))...)
    vec = []
    if type == "Int"
        vec = parse.(Int, split(ls[2]))
    elseif type == "Float32"
        vec = parse.(Float32, split(ls[2]))
    else
        println("invalid read")
        exit(-1)
    end
    if length(shape) == 1
        return copy(reshape(vec, shape))
    elseif length(shape) == 2 # Matrix: keep the number of rows&column
        return copy(reshape(vec, reverse(shape))')
    else # (>2)-dim vector: (2, 3, 4) -> (4, 3, 2)
        return copy(reshape(vec, reverse(shape)))
    end
end

function write_vec(file::String, data)
    f = open(file, "w")

    sz = size(data)
    for i in reverse(sz)
        print(f, i)
        print(f, ' ')
    end
    print(f, '\n')

    for i in data
        print(f, i)
        print(f, ' ')
    end
    return nothing
end
    