CUDA_VISIBLE_DEVICES=0,1 python train.py --backbone mobilenet --lr 0.002 --workers 4  --epochs 30 --batch-size 12 --gpu-ids 0,1 --test-batch-size 64  --eval-interval 1 --dataset vocdetection
