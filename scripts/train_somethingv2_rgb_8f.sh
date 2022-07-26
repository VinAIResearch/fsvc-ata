python -u main.py somethingv2 RGB --arch resnet50 \
--num_segments 8 --lr 0.001 --lr_steps 10 20 --epochs 25 --batch-size 32 \
-j 2 --dropout 0.5 --root_log ./checkpoints/path \
--root_model ./checkpoints/path --eval-freq=1 --npb \
--wd 0.0005 --print-freq 200 --gpus 0 --episodes 600
