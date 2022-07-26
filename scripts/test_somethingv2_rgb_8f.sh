# test on Something-Something V2
python -u main.py somethingv2 RGB --arch resnet50 --num_segments 8 --workers 2 \
--root_log ./checkpoints/path --root_model ./checkpoints/path \
--resume ./../somethingv2.path.tar --evaluate --gpus 0 --way 5 --shot 1 --episodes 10000
