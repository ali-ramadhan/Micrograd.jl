import Base: show

rand_uniform(a, b) = a + (b - a) * rand()

struct Neuron{W,B,A}
    w::W
    b::B
    σ::A

    function Neuron(n, σ=tanh)
        w = [Value(rand_uniform(-1, 1)) for _ in 1:n]
        b = Value(rand_uniform(-1, 1))
        W = typeof(w)
        B = typeof(b)
        A = typeof(σ)
        return new{W,B,A}(w, b, σ)
    end
end

(N::Neuron)(x) = N.σ(sum(N.w .* x) + N.b)

params(N::Neuron) = vcat(N.w, N.b)

show(io::IO, N::Neuron) = print(io, "Neuron: $(length(N.w)) inputs -> 1 output ($(N.σ))")

struct Layer{N}
    neurons::N

    function Layer(n_inputs, n_outputs, σ=tanh)
        neurons = [Neuron(n_inputs, σ) for _ in 1:n_outputs]
        N = typeof(neurons)
        return new{N}(neurons)
    end
end

function (L::Layer)(x)
    out = [N(x) for N in L.neurons]
    return length(out) == 1 ? out[1] : out
end

params(L::Layer) = vcat([params(N) for N in L.neurons]...)

function show(io::IO, L::Layer)
    n_inputs = length(L.neurons[1].w)
    n_outputs = length(L.neurons)
    print(io, "Layer ($n_inputs inputs -> $n_outputs outputs) with $(length(L.neurons)) neurons of type: ")
    show(io, L.neurons[1])
end

struct MultiLayerPerceptron{L}
    layers::L

    function MultiLayerPerceptron(n_inputs, n_outputs, σ=tanh)
        layer_sizes = vcat(n_inputs, n_outputs)
        n_layers = length(layer_sizes)
        layers = [Layer(layer_sizes[i], layer_sizes[i+1], σ) for i in 1:n_layers-1]
        L = typeof(layers)
        return new{L}(layers)
    end
end

function (M::MultiLayerPerceptron)(x)
    for layer in M.layers
        x = layer(x)
    end
    return x
end

params(M::MultiLayerPerceptron) = vcat([params(L) for L in M.layers]...)

function show(io::IO, M::MultiLayerPerceptron)
    n_inputs = length(M.layers[1].neurons[1].w)
    n_outputs = length(M.layers[end].neurons)
    print(io, "MultiLayerPerceptron ($n_inputs inputs -> $n_outputs outputs) with $(length(M.layers)) layers:\n")
    for (i, layer) in enumerate(M.layers)
        print(io, "  Layer $i: ")
        show(io, layer)
        i != length(M.layers) && print(io, "\n")
    end
end
