using Pkg
Pkg.activate(".")

using FractalAnimation
using CUDA: functional

function main() 

    gpu = functional()

    func = (z,c) -> tan(z)^2 + cos(z) + c
    c = 0.30+0.53im
    p = SetParams(-2.0-1.125im, 2.0+1.125im, 960, 20.0, 2160, gpu)
    γ = Path(t -> 2/3*sqrt(t) - π/5 + t*1im ,0,2)

    batched_animate(p,γ,func,"4ktest.mov") 

end

begin
    main()
end