
using Plots: plot

struct Path
    parameterization::Function
    start::Complex{Float64}
    ending::Complex{Float64}
end

CirclePath(r::Real)::Path = Path(t -> (r*ℯ^(t*1im)),0,2π)

pointsonpath(P::Path, n::Integer) = (n <= 0) ? error("n must be a positive integer") : [P.parameterization(t) for t in range(P.start, P.ending, length=n)]

function dividepath(P::Path, n::Integer)::Vector{Path}
    ending = P.ending
end