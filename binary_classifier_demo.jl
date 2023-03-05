using Statistics
using Printf

using CairoMakie
using Micrograd

# Make two interleaving half circles with `n_samples` data points in each class (0 and 1).
# See: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html
function make_moons(n_samples; noise)

    t = range(0, π, length=n_samples)
    outer_circ_x = @. cos(t)
    outer_circ_y = @. sin(t)
    inner_circ_x = @. 1 - cos(t)
    inner_circ_y = @. 1 - sin(t) - 0.5

    x = vcat(outer_circ_x, inner_circ_x) .+ noise .* randn(2n_samples)
    y = vcat(outer_circ_y, inner_circ_y) .+ noise .* randn(2n_samples)
    class = vcat(zeros(n_samples), ones(n_samples))

    return x, y, class
end

x, y, class = make_moons(100, noise=0.1)

# Make classes -1 and 1 for hinge loss.
class = @. 2*class - 1

model = MultiLayerPerceptron(2, [16, 16, 1], relu)

println(model)
println("Number of parameters: ", length(params(model)))

function loss()
    # Hinge or max-margin loss
    class_predictions = [model([x′, y′]) for (x′, y′) in zip(x, y)]
    hinge_loss = mean(relu.(1.0 .- class .* class_predictions))

    # L² regularization
    α = 1e-4
    reg_loss = α * mean([p*p for p in params(model)])

    total_loss = hinge_loss + reg_loss

    correct = count([(cp.data > 0) == (c == 1) for (cp, c) in zip(class_predictions, class)])
    accuracy = correct / length(class)

    return total_loss, hinge_loss.data, reg_loss.data, accuracy
end

η = 0.01 # learning rate
epochs = 250

for e in 1:epochs
    # forward pass
    total_loss, hinge_loss, reg_loss, accuracy = loss()

    # backward pass
    zero_gradients!(model)
    backward!(total_loss)

    # update
    for p in params(model)
        p.data += η * p.grad
    end

    @printf("Epoch %d (η=%.4f): loss = %.8f (hinge=%.8f, reg=%.8f), accuracy = %.2f%%\n", e, η(e), hinge_loss, reg_loss, total_loss.data, 100*accuracy)
end

# Plot binary classifier predictions with Makie.jl

# visualize decision boundary
N = 100
xs = range(-2, 3, length=N)
ys = range(-2, 2, length=N)

predictions = zeros(N, N)
for (i, x′) in enumerate(xs)
    for (j, y′) in enumerate(ys)
        predictions[i, j] = model([x′, y′]).data
    end
end

fig = Figure(resolution=(800, 800))
ax = Axis(fig[1, 1], xlabel="x", ylabel="y")

heatmap!(ax, xs, ys, predictions, colormap=:balance, colorrange=(-1, 1))
scatter!(ax, x[class .== -1], y[class .== -1], color=:blue)
scatter!(ax, x[class .==  1], y[class .==  1], color=:red)

save("binary_classifier_moons.png", fig, px_per_unit=2)
