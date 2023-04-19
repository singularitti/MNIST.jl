using MLDatasets: MLDatasets

export loaddata

function loaddata()
    𝗫, 𝘆 = MLDatasets.MNIST(:train)[:]
    𝗫 = (vec(X) for X in eachslice(𝗫; dims=3))  # Each original data is 28x28 matrices
    𝘆 = ([index == y for index in 0:9] for y in 𝘆)  # Each original data is a number
    return zip(𝗫, 𝘆)
end
