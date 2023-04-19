using ComputedFieldTypes: @computed

export Network, feedforward, eachlayer

const Maybe{T} = Union{T,Nothing}

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

struct EachLayer{N}
    network::N
end

eachlayer(network::Network) = EachLayer(network)

# See https://github.com/JuliaLang/julia/blob/1715110/base/strings/string.jl#L207-L213
function Base.iterate(iter::EachLayer, state=firstindex(iter))
    if state == 1
        return (first(iter.network.layers), nothing, nothing), 2
    elseif state >= length(iter)
        return nothing
    else
        return (
            iter.network.layers[state],
            iter.network.weights[state - 1],  # Note the index here!
            iter.network.biases[state - 1],  # Note the index here!
        ),
        state + 1
    end
end

Base.eltype(::EachLayer) = (Int64, Maybe{Matrix{Float64}}, Maybe{Vector{Float64}})

Base.length(iter::EachLayer) = length(size(iter))

Base.size(iter::EachLayer) = iter.network.layers
Base.size(iter::EachLayer, dim) = size(iter)[dim]

function Base.getindex(X::EachLayer, i)
    if i == 1
        return first(X.network.layers), nothing, nothing
    else
        return X.network.layers[i], X.network.weights[i - 1], X.network.biases[i - 1]
    end
end

Base.firstindex(::EachLayer) = 1

Base.lastindex(X::EachLayer) = length(X)

Base.show(io::IO, network::Network) = print(io, join(size(network), "×"), " network")
function Base.show(io::IO, ::MIME"text/plain", network::Network)
    print(io, "Network of size ", join(size(network), "×"))
    return nothing
end
