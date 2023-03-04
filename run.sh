unset CYCLES_CUDA_ADAPTIVE_COMPILE
export ASAN_OPTIONS=halt_on_error=0
build/Debug/bin/cycles --samples 100 --device OPTIX --output image.png "/media/jeremy/DRIVE/home/jeremy/Documents/Kitchen_set/Kitchen_set.usd" --width 1024 --height 512

# sudo mount /dev/nvme0n1p3 /media/jeremy/DRIVE/
