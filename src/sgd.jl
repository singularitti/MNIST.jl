using Random: shuffle

export train!

function train!(
    network::Network, data::AbstractVector{Example}, batchsize::Integer, η, nepochs=1
)
    for _ in 1:nepochs
        data = shuffle(data)
        batches = Iterators.partition(data, batchsize)
        for batch in batches
            train!(network, batch, η)
        end
    end
    return network
end
function train!(network::Network, batch::AbstractVector{Example}, η)
    η′ = η / length(batch)
    # Update each layer's weights and biases
    for example in batch
        train!(network, example, η′)
    end
    return network
end
function train!(network::Network, example::Example, η)
    𝝯w, 𝝯𝗯 = Backpropagator(sigmoid, sigmoid′)(network, example)
    for (w, 𝗯, ∇w, ∇𝗯) in zip(network.weights, network.biases, 𝝯w, 𝝯𝗯)
        w[:, :] .-= η * ∇w
        𝗯[:] .-= η * ∇𝗯
    end
    return network
end
function train(network::Network, example::Example, η)
    𝝯w, 𝝯𝗯 = Backpropagator(sigmoid, sigmoid′)(network, example)
    new_network = deepcopy(network)
    for (w, 𝗯, ∇w, ∇𝗯) in zip(new_network.weights, new_network.biases, 𝝯w, 𝝯𝗯)
        w[:, :] .-= η * ∇w
        𝗯[:] .-= η * ∇𝗯
    end
    return new_network
end
