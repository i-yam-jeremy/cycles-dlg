unset CYCLES_CUDA_ADAPTIVE_COMPILE
cd build
bin/cycles --samples 100 --device CUDA --output ../image.png ../examples/scene_monkey.xml