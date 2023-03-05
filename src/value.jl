import Base: show, +, -, *, /, ^, inv, tanh

#####
##### Value definition
#####

mutable struct Value{T, C, B}
    data::T
    grad::T
    children::C
    backward::B

    function Value(data, children)
        children = Set(children)
        D = typeof(data)
        C = typeof(children)
        return new{D,C,Function}(data, zero(D), children, () -> nothing)
    end
end

Value(data) = Value(data, ())

show(io::IO, v::Value) = print(io, "Value($(v.data), grad=$(v.grad))")

#####
#####  Operations and backpropagation rules
#####

valueify(T, x) = x isa Value ? x : Value(convert(T, x))

function +(x::Value{T}, y::Value{T}) where T
    out = Value(x.data + y.data, (x, y))

    function _backward()
        x.grad += one(T) * out.grad
        y.grad += one(T) * out.grad
    end

    out.backward = _backward

    return out
end

+(x, y::Value{T}) where T = +(valueify(T, x), y)
+(x::Value{T}, y) where T = +(x, valueify(T, y))

function -(x::Value{T}, y::Value{T}) where T
    out = Value(x.data - y.data, (x, y))

    function _backward()
        x.grad += one(T) * out.grad
        y.grad += one(T) * out.grad
    end

    out.backward = _backward

    return out
end

-(x, y::Value{T}) where T = -(valueify(T, x), y)
-(x::Value{T}, y) where T = -(x, valueify(T, y))

function *(x::Value, y::Value)
    out = Value(x.data * y.data, (x, y))

    function _backward()
        x.grad += y.data * out.grad
        y.grad += x.data * out.grad
    end

    out.backward = _backward

    return out
end

*(x, y::Value{T}) where T = *(valueify(T, x), y)
*(x::Value{T}, y) where T = *(x, valueify(T, y))

function ^(x::Value{T}, k::Number) where T
    out = Value(x.data^k, (x, ))

    function _backward()
        x.grad += convert(T, k * x.data^(k-1) * out.grad)
    end

    out.backward = _backward

    return out
end

inv(x::Value) = ^(x, -1.0)

/(x::Value, y::Value) = x * ^(y, -1)

/(x::Value{T}, y) where T = x * ^(valueify(T, y), -1)
/(x, y::Value{T}) where T = valueify(T, x) * ^(y, -1)

function tanh(x::Value)
    t = tanh(x.data)
    out = Value(t, (x, ))

    function _backward()
        x.grad += (1 - t^2) * out.grad
    end

    out.backward = _backward

    return out
end

function relu(x::Value{T}) where T
    out = Value(max(zero(T), x.data), (x, ))

    function _backward()
        x.grad += (x.data > 0) * out.grad
    end

    out.backward = _backward

    return out
end

#####
##### Backpropagation through expression tree
#####

function topological_sort(v, topo=[])
    visited = Set()
    if v ∉ visited
        for child in v.children
            topological_sort(child, topo)
        end
        push!(topo, v)
    end
    return topo
end

function backward!(v::Value{T}) where T
    v.grad = one(T)
    for node in reverse(topological_sort(v))
        node.backward()
    end
end
