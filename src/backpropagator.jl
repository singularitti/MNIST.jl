export backprop

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
