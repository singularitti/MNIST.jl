module MNIST

using ComputedFieldTypes: @computed
using Random: shuffle

export Network, feedforward, sgd!

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

end
