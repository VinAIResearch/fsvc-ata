# train on Something-Something V2
python -u main.py somethingv2 RGB --arch resnet50 \
--num_segments 8 --lr 0.001 --lr_steps 10 20 --epochs 25  \
--batch-size 32 --workers 2 --dropout 0.5 \
--root_log ./checkpoints/path --root_model ./checkpoints/path \
--wd 0.0005 --gpus 0 --episodes 600
