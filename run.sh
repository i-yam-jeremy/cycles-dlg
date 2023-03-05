unset CYCLES_CUDA_ADAPTIVE_COMPILE
export ASAN_OPTIONS=halt_on_error=0
# build/Release/bin/cycles --samples 10 --device OPTIX --output image.png "/media/jeremy/DRIVE/home/jeremy/Documents/alab-v2.0.1/ALab/entry.usda" --width 4096 --height 2048
build/Release/bin/cycles --samples 10 --device OPTIX --output image.png "/media/jeremy/DRIVE/home/jeremy/Documents/island-usd-v2.0/island/usd/island.usda" --width 4096 --height 2048
# build/Release/bin/cycles --samples 10 --device OPTIX --output image.png "/media/jeremy/DRIVE/home/jeremy/Documents/Kitchen_set/Kitchen_set_instanced.usd" --width 4096 --height 2048

# sudo mount /dev/nvme0n1p3 /media/jeremy/DRIVE/
