using Random: shuffle

export sgd!

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
    𝝯w, 𝝯𝗯 = collect(zeros(size(weights)) for weights in network.weights),
    collect(zeros(size(biases)) for biases in network.biases)
    for (x, y) in mini_batch
        𝝯wⁱ, 𝝯𝗯ⁱ = Backpropagator(sigmoid, sigmoid′)(network, x, y)
        for j in eachindex(𝝯w)
            𝝯w[j][:, :] .+= 𝝯wⁱ[j]
            𝝯𝗯[j][:] .+= 𝝯𝗯ⁱ[j]
        end
    end
    η′ = η / length(mini_batch)
    # Update each layer's weights and biases
    for (wⱼₖ, bⱼ, ∇wⱼₖ, ∇bⱼ) in zip(network.weights, network.biases, 𝝯w, 𝝯𝗯)
        wⱼₖ[:, :] .-= η′ * ∇wⱼₖ
        bⱼ[:] .-= η′ * ∇bⱼ
    end
    return network
end
