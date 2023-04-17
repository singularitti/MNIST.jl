module MNIST

using ComputedFieldTypes: @computed
using Random: shuffle
using Statistics: mean

export Network, feedforward, computecost, sgd!

@computed struct Network{N}
    layers::NTuple{N,Int64}
    weights::NTuple{N - 1,Matrix{Float64}}
    biases::NTuple{N - 1,Vector{Float64}}
end
function Network(layers)
    weights = Tuple(
        Matrix{Float64}(undef, nj, nk) for
        (nj, nk) in zip(layers[2:end], layers[1:(end - 1)])
    )
    biases = Tuple(Vector{Float64}(undef, nj) for nj in layers[2:end])
    return Network{length(layers)}(layers, weights, biases)
end
Network(layers::Integer...) = Network(layers)

function feedforward(network::Network, 𝐚)
    for (w, 𝐛) in (network.weights, network.biases)
        𝐚 = w * 𝐚 .+ 𝐛
    end
    return 𝐚
end

function computecost(network::Network, input, desired_output)
    output = feedforward(network, input)
    return sum(abs2, desired_output .- output)
end

function sgd!(network::Network, training_data, mini_batch_size, η, epochs=1)
    for _ in 1:epochs
        training_data = shuffle(training_data)
        mini_batches = Iterators.partition(training_data, mini_batch_size)
        for mini_batch in mini_batches
            update_mini_batch!(network, mini_batch, η)
        end
    end
    return network
end

function update_mini_batch!(network::Network, mini_batch, η)
    networks = map(mini_batch) do (x, y)
        backprop(network, x, y)  # For all layers
    end
    ∇w, ∇b = mean(network.weights for network in networks),
    mean(network.weights for network in networks)
    # Update each layer's weights and biases
    for (wⱼₖ, bⱼ, ∇wⱼₖ, ∇bⱼ) in zip(network.weights, network.biases, ∇w, ∇b)
        wⱼₖ[:, :] .-= η * ∇wⱼₖ
        bⱼ[:] .-= η * ∇bⱼ
    end
    return network
end

function backprop(network::Network, x, y) end

sigmoid(z) = 1 / (1 + exp(-z))

sigmoid′(z) = sigmoid(z) * (1 - sigmoid(z))

end
