using MAT

function open_data(file_name)
    mat = matread(file_name)
    return mat
end

file_name = "./WT_NoStim.mat"
data = open_data(file_name)
println(keys(data["WT_NoStim"]))