unset CYCLES_CUDA_ADAPTIVE_COMPILE
export ASAN_OPTIONS=halt_on_error=0
build/Debug/bin/cycles --samples 100 --device OPTIX --output image.png "test.usda" --width 1024 --height 512
