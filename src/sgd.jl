using Random: shuffle

export train!

function train!(network::Network, data, batch_size, η, nepochs=1)
    for _ in 1:nepochs
        training_data = shuffle(data)
        mini_batches = Iterators.partition(training_data, batch_size)
        for mini_batch in mini_batches
            update_mini_batch!(network, mini_batch, η)
        end
    end
    return network
end

function update_mini_batch!(network::Network, batch, η)
    𝝯w, 𝝯𝗯 = collect(zeros(size(weights)) for weights in network.weights),
    collect(zeros(size(biases)) for biases in network.biases)
    for (x, y) in batch
        𝝯wⁱ, 𝝯𝗯ⁱ = Backpropagator(sigmoid, sigmoid′)(network, x, y)
        for j in eachindex(𝝯w)
            𝝯w[j][:, :] .+= 𝝯wⁱ[j]
            𝝯𝗯[j][:] .+= 𝝯𝗯ⁱ[j]
        end
    end
    η′ = η / length(batch)
    # Update each layer's weights and biases
    for (wⱼₖ, bⱼ, ∇wⱼₖ, ∇bⱼ) in zip(network.weights, network.biases, 𝝯w, 𝝯𝗯)
        wⱼₖ[:, :] .-= η′ * ∇wⱼₖ
        bⱼ[:] .-= η′ * ∇bⱼ
    end
    return network
end
