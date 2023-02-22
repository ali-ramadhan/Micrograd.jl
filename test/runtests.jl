using Test
using Micrograd

@testset "Micrograd" begin
    @testset "+" begin
        a = Value(2.5)
        b = Value(-7.0)
        c = a + b

        @test a.data == 2.5
        @test b.data == -7.0
        @test c.data == -4.5

        @test isempty(a.children)
        @test isempty(b.children)
        @test a in c.children
        @test b in c.children

        @test isnothing(a.op)
        @test isnothing(b.op)
        @test c.op == +

        a = Value(-1.0)
        b = Value(4.5)
        c = a + b
        @test c.data == 3.5
    end

    @testset "*" begin
        a = Value(3//8)
        b = Value(1//8)
        c = a * b
        @test c.data == 3//64
    end

    @testset "grad" begin
        x1 = Value(2.0)
        x2 = Value(0.0)
        w1 = Value(-3.0)
        w2 = Value(1.0)
        b = Value(6.8813735870195432)

        c1 = x1 * w1
        c2 = x2 * w2
        d = c1 + c2
        n = d + b
        o = tanh(n)

        backward!(o)

        @test o.grad == 1.0
        @test n.grad ≈ 0.5

        @test d.grad ≈ 0.5
        @test b.grad ≈ 0.5

        @test c1.grad ≈ 0.5
        @test c2.grad ≈ 0.5

        @test x1.grad ≈ -1.5
        @test w1.grad ≈ 1.0

        @test x2.grad ≈ 0.5
        @test w2.grad ≈ 0
    end

    @testset "MultiLayerPerceptron" begin
        xs = [
            [2.0, 3.0, -1.0],
            [3.0, -1.0, 0.5],
            [0.5, 1.0, 1.0],
            [1.0, 1.0, -1.0]
        ]

        ys = [1.0, -1.0, -1.0, 1.0]

        MLP = MultiLayerPerceptron(3, [4, 4, 1])

        epochs = 200
        ϵ = 0.05

        for e in 1:epochs
            # forward pass
            y_pred = [MLP(x) for x in xs]
            loss = sum([(y - ŷ)^2 for (y, ŷ) in zip(ys, y_pred)])

            # backward pass
            for p in params(MLP)
                p.grad = 0
            end
            backward!(loss)

            # update
            for p in params(MLP)
                p.data += ϵ * p.grad
            end

            new_y_pred = [MLP(x) for x in xs]
            new_loss = sum([(y - ŷ)^2 for (y, ŷ) in zip(ys, new_y_pred)])

            @show e, loss.data

            if e > 1
                @test new_loss.data < loss.data
            end
        end

    end
end
