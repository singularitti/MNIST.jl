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
    batchsize = length(batch)
    new_networks = collect(train(network, example, η / batchsize) for example in batch)
    new_weights = sum(new_network.weights for new_network in new_networks) / batchsize
    new_biases = sum(new_network.biases for new_network in new_networks) / batchsize
    for (weight, bias, new_weight, new_bias) in
        zip(network.weights, network.biases, new_weights, new_biases)
        weight[:] = new_weight
        bias[:] = new_bias
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
