
"""
    flatten 2D matrix into (m*n, 1) matrix

        mainly used for images to flatten images

    inputs:
        x := 3D (rgp, m, n) matrix

    outputs:
        y := 2D (rgp*m*n, 1) matrix
"""
function flatten(x)
    rgp, n, m = size(x)
    return reshape(x, (rgp*m*n, 1))
end
