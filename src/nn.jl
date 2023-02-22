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

struct Layer{N}
    neurons::N

    function Layer(n_inputs, n_outputs)
        neurons = [Neuron(n_inputs) for _ in 1:n_outputs]
        N = typeof(neurons)
        return new{N}(neurons)
    end
end

function (L::Layer)(x)
    out = [N(x) for N in L.neurons]
    return length(out) == 1 ? out[1] : out
end

params(L::Layer) = vcat([params(N) for N in L.neurons]...)

struct MultiLayerPerceptron{L}
    layers::L

    function MultiLayerPerceptron(n_inputs, n_outputs)
        layer_sizes = vcat(n_inputs, n_outputs)
        n_layers = length(layer_sizes)
        layers = [Layer(layer_sizes[i], layer_sizes[i+1]) for i in 1:n_layers-1]
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
