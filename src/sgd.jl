using Random: shuffle
using Statistics: mean

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

function backprop(network::Network, f, f′, 𝘅, 𝘆)
    iter = Iterators.rest(eachlayer(network))  # Start from the first hidden layer
    # Feed forward
    zs, activations = Vector[], [𝘅]
    𝗮 = 𝘅
    for (_, wˡ, 𝗯ˡ) in iter
        𝘇ˡ = wˡ * 𝗮 .+ 𝗯ˡ
        push!(zs, 𝘇ˡ)
        𝗮 = f.(𝘇ˡ)
        push!(activations, 𝗮)
    end
    𝘇ᴸ, 𝗮ᴸ = zs[end], activations[end]
    # Backward pass
    𝝳 = (𝗮ᴸ .- 𝘆) .* f′.(𝘇ᴸ)  # 𝝳ᴸ
    𝝯w, 𝝯𝗯 = [𝝳 .* activations[end - 1]], [𝝳]
    for ((_, wˡ⁺¹, _), 𝘇ˡ, 𝗮ˡ⁻¹) in
        Iterators.reverse(zip(iter, zs, activations[begin:(end - 1)]))
        𝝳 = transpose(wˡ⁺¹) * 𝝳 .* f′.(𝘇ˡ)
        push!(𝝯w, 𝝳 .* 𝗮ˡ⁻¹)
        push!(𝝯𝗯, 𝝳)
    end
    return 𝝯w, 𝝯𝗯
end

sigmoid(z) = 1 / (1 + exp(-z))

sigmoid′(z) = sigmoid(z) * (1 - sigmoid(z))
