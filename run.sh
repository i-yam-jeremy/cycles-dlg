unset CYCLES_CUDA_ADAPTIVE_COMPILE
cd build/Release
bin/cycles --samples 100 --device OPTIX --output ../../image.png ../../examples/scene_monkey.xml