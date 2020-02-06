a = rand(20, 5);
b = rand(13, 5);

try
    a .+ b
catch DimensionMismatch
    
