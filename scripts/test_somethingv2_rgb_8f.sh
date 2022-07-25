# test on Something-Something V2
python -u main.py somethingv2 RGB --arch resnet50 --num_segments 8 -j 2 \
--root_log ./checkpoints/path --root_model ./checkpoints/path --npb \
--resume ./../somethingv2.path.tar --evaluate --gpus 0 --iter 50 --shot 1
