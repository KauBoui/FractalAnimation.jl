
struct Path
    parameterization::Function
    start::Complex{Float64}
    ending::Complex{Float64}
end

CirclePath(r::Real)::Path = Path(t -> (r*ℯ^(t*1im)),0,2π)

pointsonpath(P::Path, n::Integer) = [P.parameterization(t) for t in range(P.start, P.ending, length=n)]
