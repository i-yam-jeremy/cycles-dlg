unset CYCLES_CUDA_ADAPTIVE_COMPILE
export ASAN_OPTIONS=halt_on_error=0
build/Release/bin/cycles --samples 10 --device OPTIX --output image.png "/media/jeremy/DRIVE/home/jeremy/Documents/island-usd-v2.0/island/usd/island.usda" --width 1024 --height 512

# sudo mount /dev/nvme0n1p3 /media/jeremy/DRIVE/
