CUDA_VISIBLE_DEVICES=0,1 python train.py --backbone resnet --lr 0.007 --workers 4  --epochs 100 --batch-size 16 --gpu-ids 0,1 --checkname deeplab-resnet --eval-interval 1 --dataset pascal
