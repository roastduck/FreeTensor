
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
    elseif length(shape) == 2
        return copy(reshape(vec, reverse(shape))')
    end
    println("invalid read")
    exit(-1)
end

function write_vec(file::String, data)
    f = open(file, "w")

    if length(size(data)) == 2
        data = copy(data')
    end

    sz = size(data)
    for i in sz
        print(f, i)
        print(f, ' ')
    end
    print(f, '\n')

    if length(size(data)) == 2
        data = copy(data')
    end

    for i in data
        print(f, i)
        print(f, ' ')
    end
    return nothing
end
    