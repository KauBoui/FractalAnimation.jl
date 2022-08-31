module FractalAnimation

using Plots

include("paths.jl")

export Path, CirclePath, pointsonpath, SetParams, juliaset, juliaprogression, mandelbrotset

"""
    Evaluates the divergence speed for a given function of z,c in the complex plane. 
    For julia and fatou sets pass the whole complex plane as zinit
    For mandelbrot-esque sets pass the whole compelx plane as c
"""
function escapeeval(f::Function, 
                    threshold::Real, 
                    c::Union{Complex, Matrix{Complex}} = 1,
                    z::Union{Complex,Matrix{Complex}} = 0, 
                    maxiter::Integer = 255) :: Int64
    for i = 1:maxiter
        z = f(z, c)
        if abs(z) >= threshold
            return i
        end
    end
    return -0
end

function _setdims(max_coord, min_coord, resolution) :: Int64
    dim = (max_coord - min_coord) * resolution
    dim == 0 ? error("Height or width cannot be 0!") : return dim
end

"""
Essentially meshgrid to produce a Complex plane array of the given size
"""

function _genplane(min_coord::Complex, max_coord::Complex, width::Int, height::Int)
    real = range(min_coord.re, max_coord.re,length=width)
    imag = range(min_coord.im, max_coord.im,length=height)
    complexplane = zeros(Complex{Float64},(width, height))
    for (i,x) ∈ collect(enumerate(real))
        complexplane[:,i] .+= x
    end
    for (i,y) ∈ collect(enumerate(imag))
        complexplane[i,:] .+= (y * 1im)
    end
    return reverse!(complexplane, dims=1)
end

struct SetParams
    min_coord::Complex{Float64}
    max_coord::Complex{Float64}
    resolution::Int64
    width::Int64
    height::Int64
    plane::Matrix{Complex{Float64}}
    threshold::Float64 
    nr_frames::Int64
    """ 
        The resolution, minimum, & maximum coordiantes determine the width & height
    """
    function SetParams(min_coord::Complex, max_coord::Complex, resolution::Integer, threshold::Real, nr_frames::Integer) 
        if min_coord.re >= max_coord.re; error("Max real component cannot be less than or equal to Min real component!") end
        if min_coord.im >= max_coord.im; error("Max imaginary component cannot be less than or equal to Min imaginary component!") end
        width = _setdims(max_coord.re, min_coord.re, resolution)
        height = _setdims(max_coord.im, min_coord.im, resolution) 
        plane = _genplane(min_coord, max_coord, width, height)
        return new( min_coord, 
            max_coord, 
            resolution, 
            width,
            height,
            plane,  
            threshold,
            nr_frames) 
    end
end 

function mandelbrotset(set_p::SetParams, f::Function, z::Complex = 0, maxiter::Integer = 255)
    return escapeeval.(f, set_p.threshold, set_p.plane, z, maxiter)
end

function juliaset(set_p::SetParams, f::Function, c::Complex, maxiter::Integer = 255)
    return escapeeval.(f, set_p.threshold, c, set_p.plane, maxiter)
end

function juliaprogression(set_p::SetParams, P::Path, f::Function, maxiter::Integer = 255)
    c_vec = pointsonpath(P,sep_p.nr_frames)
    return [juliaset(set_p, c, f, maxiter) for c ∈ c_vec]
end 

function animateprogression(sets::Vector{AbstractArray}, set_p::SetParams, colormap=:terrain, file_name::String ="~/GIFs/julia_set.gif")
    anim = @animate for set ∈ sets
        heatmap(set, size=(set_p.width,set_p.height), color=colormap, leg=false)
    end
    gif(anim, file_name, fps=30)
end

end # module
