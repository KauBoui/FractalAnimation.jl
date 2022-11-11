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

    p_to_gpu = params |> to_gpu!

    @test params.gpu = true
    @test p_to_gpu = true
    
    # ---- Path tests ---- # 

    sin_path = Path(t-> t + 2im*sin(t),0,2π)

    @test sin_path.parameterization(0) ≈ 0 + 0im atol=0.01
    @test sin_path.parameterization(2π) ≈ 2π + 0im atol=0.01
    @test pointsonpath(sin_path,2) ≈ [0.0+0.0im, 2π + 0im] atol=0.01

    @test_throws ErrorException pointsonpath(sin_path, 0)
    @test_throws ErrorException pointsonpath(sin_path, -2)

    test_params = SetParams(-1.0-1.0im, 1.0+1.0im, 10, 20.0, 4)
    func = (z,c) -> z^2 + c
    c = 0.30+0.53im

    correct_result_julia =    [ 5   6   0  16  13   8   7   7   7   8  13  37   8   5   5   4   4   4   3  3 ;
                                5   7  11   0   0  12   0   9   9   0   0   0   8   6   5   4   4   4   3  3 ;
                                8  23   0   0   0   0   0  18  12   0   0   0   9   7   6   5   4   4   4  3 ;
                                0  16   0   0   0   0   0   0  70  32   0  15  15   9   8   5   4   4   4  4 ;
                                0   8   9   9  15   0   0   0   0  30  39   0   0  20  10   6   5   4   4  4 ;
                                6   6   7   8  11  24  28   0   0   0   0  23   0  13   8   6   5   4   4  4 ;
                                5   5   6   7   8  12  11  14  16   0   0  35  11   9   7   6   5   4   4  4 ;
                                5   5   5   6   7   8   9  12   0   0   0   0  14   9   7   6   5   5   4  4 ;
                                4   5   5   6   6   7   9   0   0   0   0   0  15  25   8   6   5   5   4  4 ;
                                4   5   5   5   6   7   0   0   0   0   0   0   0   0   8   6   5   5   4  4 ;
                                4   4   5   5   6   8   0   0   0   0   0   0   0   0   7   6   5   5   5  4 ;
                                4   4   5   5   6   8  25  15   0   0   0   0   0   9   7   6   6   5   5  4 ;
                                4   4   5   5   6   7   9  14   0   0   0   0  12   9   8   7   6   5   5  5 ;
                                4   4   4   5   6   7   9  11  35   0   0  16  14  11  12   8   7   6   5  5 ;
                                4   4   4   5   6   8  13   0  23   0   0   0   0  28  24  11   8   7   6  6 ;
                                4   4   4   5   6  10  20   0   0  39  30   0   0   0   0  15   9   9   8  0 ;
                                4   4   4   4   5   8   9  15  15   0  32  70   0   0   0   0   0   0  16  0 ;
                                3   4   4   4   5   6   7   9   0   0   0  12  18   0   0   0   0   0  23  8 ;
                                3   3   4   4   4   5   6   8   0   0   0   9   9   0  12   0   0  11   7  5 ;
                                3   3   4   4   4   5   5   8  37  13   8   7   7   7   8  13  16   0   6  5 ]

    @test julia_set(test_params, func, c) == correct_result_julia

    correct_result_mandelbrot =   [ 5   5   5   5   6   6   7   8  12  12   7   6   5   5  5  5  4  4  4  4 ;
                                    5   5   6   6   6   7   7  10  41  14   8   7   6   6  5  5  5  4  4  4 ;
                                    5   6   6   6   7   8   9  14   0   0  10   8   7   6  6  5  5  4  4  4 ;
                                    6   6   7   8  11  10  13  15   0   0  13  11   9  12  6  5  5  5  4  4 ;
                                    7   7   7   9  89   0   0   0   0   0   0  38  34  19  7  6  5  5  4  4 ;
                                    7   8   8  16  34   0   0   0   0   0   0   0   0  11  7  6  5  5  4  4 ;
                                   11   9  10  23   0   0   0   0   0   0   0   0   0  28  9  6  5  5  5  4 ;
                                    0  19  13   0   0   0   0   0   0   0   0   0   0   0  8  6  5  5  5  4 ;
                                    0   0  35   0   0   0   0   0   0   0   0   0   0   0  8  6  5  5  5  4 ;
                                    0   0   0   0   0   0   0   0   0   0   0   0   0  10  7  6  5  5  5  4 ;
                                    0   0   0   0   0   0   0   0   0   0   0   0   0  10  7  6  5  5  5  4 ;
                                    0   0  35   0   0   0   0   0   0   0   0   0   0   0  8  6  5  5  5  4 ;
                                    0  19  13   0   0   0   0   0   0   0   0   0   0   0  8  6  5  5  5  4 ;
                                   11   9  10  23   0   0   0   0   0   0   0   0   0  28  9  6  5  5  5  4 ;
                                    7   8   8  16  34   0   0   0   0   0   0   0   0  11  7  6  5  5  4  4 ;
                                    7   7   7   9  89   0   0   0   0   0   0  38  34  19  7  6  5  5  4  4 ;
                                    6   6   7   8  11  10  13  15   0   0  13  11   9  12  6  5  5  5  4  4 ;
                                    5   6   6   6   7   8   9  14   0   0  10   8   7   6  6  5  5  4  4  4 ;
                                    5   5   6   6   6   7   7  10  41  14   8   7   6   6  5  5  5  4  4  4 ;
                                    5   5   5   5   6   6   7   8  12  12   7   6   5   5  5  5  4  4  4  4 ; ]

    @test mandelbrot_set(test_params, func, c) == correct_result_mandelbrot
    
end