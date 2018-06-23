#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/tf_models:/tf_models/research:/tf_models/research/slim
export PATH=$PATH:$PYTHONPATH
python ../tf_models/research/setup.py build
python ../tf_models/research/setup.py install
python ../tf_models/research/object_detection/builders/model_builder_test.py
python /tf_models/research/object_detection/train.py --logtostderr --train_dir=/output/ --pipeline_config_path=./faster_rcnn.config
#echo $PATH
#ls ../
