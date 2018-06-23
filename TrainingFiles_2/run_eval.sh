#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/tf_models:/tf_models/research:/tf_models/research/slim
export PATH=$PATH:$PYTHONPATH
python ../tf_models/research/setup.py build
python ../tf_models/research/setup.py install

pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"

cp -R /model/* /output
python /tf_models/research/object_detection/eval.py --logtostderr --checkpoint_dir=/output/ --eval_dir=/output/ --pipeline_config_path=./faster_rcnn.config
#echo $PATH
#ls ../
