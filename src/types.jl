using ComputedFieldTypes: @computed

export Network, feedforward

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

function (network::Network)(f, 𝘅, 𝘆)
    𝘆̂ = feedforward(f, network.weights, network.biases, 𝘅)
    return sum(abs2, 𝘆 .- 𝘆̂)
end

function feedforward(f, weights, biases, 𝗮)
    for (w, 𝗯) in zip(weights, biases)
        𝗮 = f.(w * 𝗮 .+ 𝗯)
    end
    return 𝗮
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
