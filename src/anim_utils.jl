using Images
using Plots: Animation, plot!, heatmap
using Printf: @sprintf

import Plots: frame

struct ImageWrapper
    img::Matrix{RGB}
end

function frame(anim::Animation, iw::ImageWrapper)
    i = length(anim.frames) + 1
    filename = @sprintf("%06d.png", i)
    save(joinpath(anim.dir,filename), iw.img)
    push!(anim.frames, filename)
end

function gen_animation(images::Vector{Matrix{RGB}})::Animation
    anim = Animation()
    wrapped_images = ImageWrapper.(images)
    map((x)->frame(anim, x), wrapped_images)
    return anim
end

get_maxval(sets::Vector{AbstractArray})::Int = maximum(map(x -> (Iterators.flatten(x) |> collect |> maximum), sets)) |> Int

apply_colorscheme(csheme::ColorScheme, sets::Vector{AbstractArray}, maxval::Int)::Vector{Matrix{RGB}} = map((x) -> get(csheme, x, (0, maxval)), sets)

complex_euclidean_distance(u::Complex, v::Complex) = sqrt((u.re - v.re)^2 + (u.im - v.im)^2)

function find_location(val::Complex, plane::AbstractArray)
    distances = complex_euclidean_distance.(val, plane)
    return findmin(distances)[2]
end

map_points_to_plane(points::Vector, plane::AbstractArray)::AbstractArray = map((point) -> find_location(point,plane), points) .|> Tuple .|> reverse
