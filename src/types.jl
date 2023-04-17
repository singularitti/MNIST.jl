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
