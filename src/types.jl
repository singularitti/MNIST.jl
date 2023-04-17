using ComputedFieldTypes: @computed

export Network

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

Base.eltype(::Network{N}) where {N} = (Int64, Matrix{Float64}, Vector{Float64})

Base.length(::Network{N}) where {N} = N

Base.size(network::Network) = size(network.layers)
