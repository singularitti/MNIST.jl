using ComputedFieldTypes: @computed

export Network, feedforward, computecost

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

function (network::Network)(input, desired_output)
    return computecost(network, input, desired_output)
end

function feedforward(network::Network, f, 𝐚)
    for (w, 𝐛) in (network.weights, network.biases)
        𝐚 = f.(w * 𝐚 .+ 𝐛)
    end
    return 𝐚
end

function computecost(network::Network, input, desired_output)
    output = feedforward(network, input)
    return sum(abs2, desired_output .- output)
end

Base.iterate(network::Network) = (
    (
        first(network.layers),
        zeros(size(first(network.weights))),
        zeros(size(first(network.biases))),
    ),
    1,
)
function Base.iterate(network::Network, state)
    if state >= length(network)
        return nothing
    else
        return (
            (network.layers[state], network.weights[state - 1], network.biases[state - 1]),  # Note the index here!
            state + 1,
        )
    end
end

Base.eltype(::Network) = (Int64, Matrix{Float64}, Vector{Float64})

Base.length(network::Network) = length(size(network))

Base.size(network::Network) = network.layers

function Base.getindex(network::Network, i)
    if i == 1
        return first(network.layers),
        zeros(size(first(network.weights))),
        zeros(size(first(network.biases)))
    else
        return network.layers[i], network.weights[i - 1], network.biases[i - 1]
    end
end

Base.firstindex(::Network) = 1

Base.lastindex(network::Network) = length(network)

Base.show(io::IO, network::Network) = print(io, join(size(network), "×"), " network")
function Base.show(io::IO, ::MIME"text/plain", network::Network)
    println(io, "Network of size ", join(size(network), "×"))
    return nothing
end