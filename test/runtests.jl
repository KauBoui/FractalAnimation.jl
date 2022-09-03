using FractalAnimation
using Test

@testset "FractalAnimation.jl" begin

    # ---- Set Parameters bounds checking ---- # 

    @test_throws ErrorException SetParams(-2.0-2.0im, 2.0+2.0im, 0, 20.0, 60)
    @test_throws ErrorException SetParams(-2.0+2.0im, 2.0+2.0im, 20, 20.0, 60)
    @test_throws ErrorException SetParams(2.0-2.0im, 2.0+2.0im, 20, 20.0, 60)
    @test_throws ErrorException SetParams(2.0+2.0im, 2.0+2.0im, 0, 20.0, 60)

    params = SetParams(-2.0-2.0im, 2.0+2.0im, 200, 20.0, 360)
    gpu_params = SetParams(-2.0-2.0im, 2.0+2.0im, 200, 20.0, 360) |> to_gpu!
    @test params.plane isa Matrix{Complex{Float64}}
    @test gpu_params.plane isa CuArray{Complex{Float64},2, CUDA.Mem.DeviceBuffer}
    
    # ---- Path tests ---- # 

    sin_path = Path(t-> t + 2im*sin(t),0,2π)

    @test sin_path.parameterization(0) ≈ 0 + 0im atol=0.01
    @test sin_path.parameterization(2π) ≈ 2π + 0im atol=0.01
    @test pointsonpath(sin_path,2) ≈ [0.0+0.0im, 2π + 0im] atol=0.01

    @test_throws ErrorException pointsonpath(sin_path, 0)
    @test_throws ErrorException pointsonpath(sin_path, -2)

end