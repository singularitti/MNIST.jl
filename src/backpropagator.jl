export Backpropagator

struct Backpropagator{F,F′}
    f::F
    f′::F′
end

function (b::Backpropagator)(network::Network, 𝘅, 𝘆)
    # Feed forward
    zs, activations = Vector{Float64}[], Vector{Float64}[𝘅]
    𝗮 = 𝘅
    for (_, wˡ, 𝗯ˡ) in excludeinput(eachlayer(network))
        𝘇ˡ = wˡ * 𝗮 .+ 𝗯ˡ
        push!(zs, 𝘇ˡ)
        𝗮 = b.f.(𝘇ˡ)
        push!(activations, 𝗮)
    end
    𝘇ᴸ, 𝗮ᴸ = zs[end], activations[end]
    # Backward pass
    𝝳 = (𝗮ᴸ .- 𝘆) .* b.f′.(𝘇ᴸ)  # 𝝳ᴸ
    𝝯w, 𝝯𝗯 = [kron(𝝳, activations[end - 1]')], [𝝳]  # 𝝯wᴸ, 𝝯𝗯ᴸ
    # Select `network` from layer L to 3, `zs` from layer L-1 to 2, `activations` from layer L-2 to 1
    for ((_, wˡ⁺¹, _), 𝘇ˡ, 𝗮ˡ⁻¹) in zip(
        Iterators.reverse(excludeinput(eachlayer(network))),
        zs[(end - 1):-1:begin],
        activations[(end - 2):-1:begin],
    )
        𝝳 = transpose(wˡ⁺¹) * 𝝳 .* b.f′.(𝘇ˡ)
        push!(𝝯w, kron(𝝳, 𝗮ˡ⁻¹'))
        push!(𝝯𝗯, 𝝳)
    end
    return reverse(𝝯w), reverse(𝝯𝗯)
end

sigmoid(z) = 1 / (1 + exp(-z))

sigmoid′(z) = sigmoid(z) * (1 - sigmoid(z))
