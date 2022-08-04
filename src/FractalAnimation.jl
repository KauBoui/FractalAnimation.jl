module FractalAnimation

using Base.Threads

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
    @threads for i = 1:maxiter
        z = f(z, c)
        if abs(z) >= threshold
            return i
        end
    end
    return -0
end

function setdims(max_coord, min_coord, resolution) :: Int64
    dim = (max_coord - min_coord) * resolution
    dim == 0 ? error("Height or width cannot be 0!") : return dim
end

struct SetParams
    min_coord::Complex{Float64}
    max_coord::Complex{Float64}
    resolution::Int64
    width::Int64
    height::Int64
    threshold::Float64 
    nr_frames::Int64
    """ 
        The resolution, minimum, & maximum coordiantes determine the width & height
    """
    function SetParams(min_coord::Complex, max_coord::Complex, resolution::Integer, threshold::Real, nr_frames::Integer) 
        if min_coord.re >= max_coord.re; error("Max real component cannot be less than or equal to Min real component!") end
        if min_coord.im >= max_coord.im; error("Max imaginary component cannot be less than or equal to Min imaginary component!") end
        return new( min_coord, 
            max_coord, 
            resolution, 
            setdims(max_coord.re, min_coord.re, resolution), 
            setdims(max_coord.im, min_coord.im, resolution), 
            threshold,
            nr_frames) 
    end
end 
"""
Essentially meshgrid to produce a Complex plane array of the given size
"""
function _genplane(set_p::SetParams)
    real = range(set_p.min_coord.re, set_p.max_coord.re,length=set_p.width)
    imag = range(set_p.min_coord.im, set_p.max_coord.im,length=set_p.height)
    complexplane = zeros(Complex{Float64},(set_p.width,set_p.height))
    @threads for (i,x) ∈ enumerate(real)
        complexplane[:,i] .+= x
    end
    @threads for (i,y) ∈ enumerate(imag)
        complexplane[i,:] .+= (y * 1im)
    end
    return reverse!(complexplane, dims=1)
end

function mandelbrotset(set_p::SetParams, f::Function, z::Complex = 0, maxiter::Integer = 255)
    plane = _genplane(set_p)
    return escapeeval.(f, set_p.threshold, plane, z, maxiter)
end

function juliaset(set_p::SetParams, c, f::Function, maxiter::Integer = 255)
    plane = _genplane(set_p)
    return escapeeval.(f, set_p.threshold, c, plane, maxiter)
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
