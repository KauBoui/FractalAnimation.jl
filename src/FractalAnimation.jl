module FractalAnimation

using Plots: gif
using ProgressMeter: @showprogress
using ColorSchemes
using VideoIO
using CUDA

include("paths.jl")
include("anim_utils.jl")

export Path, CirclePath, pointsonpath, SetParams, juliaset, juliaprogression, animateprogression, show_mandelbrot_traversal, mandelbrotset, to_gpu, batched_animate

"""
    Evaluates the divergence speed for a given function of z,c in the complex plane. 
    For julia and fatou sets pass the whole complex plane as z
    For mandelbrot-esque sets pass the whole complex plane as c
"""
function escapeeval(f::Function, 
                    threshold::Real, 
                    c::Union{Complex, Matrix{Complex}} = 1,
                    z::Union{Complex, Matrix{Complex}} = 0, 
                    maxiter::Integer = 255)
    for i = 1:maxiter
        z = f(z, c)
        if abs(z) ≥ threshold
            return i
        end
    end
    return -0
end

function _setdims(max_coord, min_coord, resolution)
    dim = (max_coord - min_coord) * resolution
    dim == 0 ? error("Height or width cannot be 0!") : return ceil(dim) |> Int64
end

"""
Essentially meshgrid to produce a Complex plane array of the given size
"""

function _genplane(min_coord::Complex, max_coord::Complex, width::Integer, height::Integer, gpu::Bool)
    real = range(min_coord.re, max_coord.re,length=width)
    imag = range(min_coord.im, max_coord.im,length=height)
    complexplane = zeros(Complex{Float64},(height, width))
    for (i,x) ∈ collect(enumerate(real))
        complexplane[:,i] .+= x
    end
    for (i,y) ∈ collect(enumerate(imag))
        complexplane[i,:] .+= (y * 1im)
    end
    reverse!(complexplane, dims=1)
    gpu == true ? (return cu(complexplane)) : (return complexplane) 
end

struct SetParams
    min_coord::Complex{Float64}
    max_coord::Complex{Float64}
    resolution::Int64
    width::Int64
    height::Int64
    plane::Union{Matrix{Complex{Float64}},CuArray{Complex{Float32}}}
    threshold::Float64 
    nr_frames::Int64
    gpu::Bool
    """ 
        The resolution, minimum, & maximum coordiantes determine the width & height
    """
    function SetParams(min_coord::Complex, max_coord::Complex, resolution::Integer, threshold::Real, nr_frames::Integer, gpu::Bool = false) 
        if min_coord.re ≥ max_coord.re; error("Max real component cannot be less than or equal to Min real component!") end
        if min_coord.im ≥ max_coord.im; error("Max imaginary component cannot be less than or equal to Min imaginary component!") end
        width = _setdims(max_coord.re, min_coord.re, resolution)::Int64
        height = _setdims(max_coord.im, min_coord.im, resolution)::Int64 
        plane = _genplane(min_coord, max_coord, width, height, gpu)::Union{Matrix{Complex{Float64}},CuArray{Complex{Float32}}}
        return new( min_coord, 
            max_coord, 
            resolution, 
            width,
            height,
            plane,  
            threshold,
            nr_frames, gpu) 
    end
end 

# ! Returns a NEW SetParams allocated on GPU
# ! This function should only be used for convenience
# ! For initial construction pass true to gpu value

function to_gpu(p::SetParams)
    if p.gpu == true
        return p
    else
        p = SetParams(p.min_coord, p.max_coord, p.resolution, p.threshold, p.nr_frames, true)
        return p
    end
end

mandelbrotset(set_p::SetParams, f::Function, z::Complex = 0.0+0.0im, maxiter::Integer = 255) = 
    set_p.gpu == true ? exec_gpu_kernel_mandelbrot(set_p, f, z, maxiter) |> Array : escapeeval.(f, set_p.threshold, set_p.plane, z, maxiter)

juliaset(set_p::SetParams, f::Function, c::Complex, maxiter::Integer = 255) = 
    set_p.gpu == true ? exec_gpu_kernel_julia(set_p, f, c, maxiter) |> Array : escapeeval.(f, set_p.threshold, c, set_p.plane, maxiter)
    
""" ------- Progression Functions ------- """

function juliaprogression(set_p::SetParams, γ::Path, f::Function, maxiter::Integer = 255) 
    points = pointsonpath(γ,set_p.nr_frames)
    return [(c,juliaset(set_p, f, c, maxiter)) for c ∈ points]
end

juliaprogression(set_p::SetParams, points::Vector, f::Function, maxiter::Integer = 255) = [(c,juliaset(set_p, f, c, maxiter)) for c ∈ points]

"""
    For a given path (γ), overlay it on top of the mandelbrot set for the given function (f)
    This shows how a julia progression of said function and path.
"""

function show_mandelbrot_traversal(set_p::SetParams, γ::Path, f::Function; heat_c=:terrain, line_c=:red, num_points::Integer=512)
    plane = set_p.plane |> Array
    points = pointsonpath(γ,num_points)
    mapped_points = map_points_to_plane(points, plane)
    mandelset = mandelbrotset(set_p, f) 
    begin
        heatmap(mandelset, size=(set_p.width, set_p.height), c=heat_c, leg=false)
        plot!(mapped_points, size=(set_p.width, set_p.height), color=line_c)
    end
end

function animateprogression(progression::Vector, cscheme=ColorSchemes.terrain)

    sets = [set for (_,set) ∈ progression]

    max = get_maxval(sets)
    images = apply_colorscheme(cscheme, sets, max)
    anim = gen_animation(images)

    return anim
end

function batched_animate(set_p::SetParams, γ::Path, func::Function, 
                         filepath::String = "progression.mp4", cscheme = ColorSchemes.terrain, 
                         batchsize::Integer = 30, maxiter::Integer = 255)
    
    encoder_options = (crf=0, preset="ultrafast")

    points = pointsonpath(γ, set_p.nr_frames)
    pointsets = Iterators.partition(points,batchsize) .|> Vector

    open_video_out(filepath, RGB{N0f8}, (set_p.height,set_p.width), framerate=60, encoder_options=encoder_options) do writer
        @showprogress "Processing Batches..." for pointset ∈ pointsets

            progression = juliaprogression(set_p, pointset, func, maxiter)
            sets = [set for (_,set) ∈ progression]
            println(typeof(sets[1]))
            imgstack = apply_colorscheme(cscheme, sets, maxiter)

            for img ∈ imgstack
                write(writer,img)
            end
        end
    end
    return nothing
end

""" ------- GPU Related Functions ------- """


"""
    CUDA kernel for julia sets
    Algorithm from: https://github.com/vini-fda/Mandelbrot-julia/blob/main/src/Mandelbrot_gpu.ipynb
"""

function kernel_julia_gpu!(out, in, f::Function, c::Complex, threshold::Real, maxiter::Integer)
	id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    w, h = size(in)

    cind = CartesianIndices((w, h))

    for k=id:stride:w*h
        i = cind[k][1]
        j = cind[k][2]
        z = in[i,j]
        itrs = 0
        while CUDA.abs2(z) < threshold 
            if itrs ≥ maxiter
                itrs = 0
                break
            end
           z = f(z,c)
           itrs += 1
        end
        @inbounds out[i,j] = itrs
    end

    return nothing
end

"""
    Wrapper for CUDA kernel
"""

function exec_gpu_kernel_julia(set_p::SetParams, f::Function, c::Complex, maxiter::Integer=255)
    plane_trace = CuArray{ComplexF32}(undef, set_p.height, set_p.width)
    out_trace = CuArray{Float32}(undef, set_p.height, set_p.width)

    kernel = @cuda name="juliaset" launch=false kernel_julia_gpu!(out_trace, plane_trace, f, c, set_p.threshold, maxiter)
    config = launch_configuration(kernel.fun)
    threads = Base.min(length(out_trace), config.threads)
    blocks = cld(length(out_trace), threads) 
  
    out = CUDA.zeros(Float32,(set_p.plane |> size)...)

    CUDA.@sync kernel(out, set_p.plane, f, c, set_p.threshold, maxiter; threads=threads, blocks=blocks)
    
    return out
end

"""
    CUDA kernel for mandelbrot sets
"""

function kernel_mandelbrot_gpu!(out, in, f::Function, z_init::Complex, threshold::Real, maxiter::Integer)
	id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    w, h = size(in)

    cind = CartesianIndices((w, h))

    for k = id:stride:w*h
        i = cind[k][1]
        j = cind[k][2]
        c = in[i,j]
        z = z_init
        itrs = 0
        while CUDA.abs2(z) < threshold 
            if itrs ≥ maxiter
                itrs = 0
                break
            end
           z = f(z,c)
           itrs += 1
        end
        @inbounds out[i,j] = itrs
    end

    return nothing
end

"""
    Wrapper for CUDA kernel 
"""

function exec_gpu_kernel_mandelbrot(set_p::SetParams, f::Function, z_init::Complex = 0, maxiter::Integer=255)
    plane_trace = CuArray{ComplexF32}(undef, set_p.height, set_p.width)
    out_trace = CuArray{Float32}(undef, set_p.height, set_p.width)

    kernel = @cuda name="juliaset" launch=false kernel_mandelbrot_gpu!(out_trace, plane_trace, f, z_init, set_p.threshold, maxiter)
    config = launch_configuration(kernel.fun)
    threads = Base.min(length(out_trace), config.threads)
    blocks = cld(length(out_trace), threads) 

    out = cu(zeros(set_p.plane |> size))

    CUDA.@sync kernel(out, set_p.plane, f, z_init, set_p.threshold, maxiter; threads=threads, blocks=blocks)
    
    return out
end

end # module