module Micrograd

export Value, backward!, relu
export Neuron, Layer, MultiLayerPerceptron, params, zero_gradients!

include("value.jl")
include("nn.jl")

end # module Micrograd
