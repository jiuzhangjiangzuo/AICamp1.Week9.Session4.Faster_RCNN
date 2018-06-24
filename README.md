# AICamp1.Week9.Session4.Faster_RCNN

## Data path

Wrong data: zhouji1994/datasets/widerface/1

Fine tune check point: zhouji1994/datasets/widerface/1

Correct data: zhouji1994/projects/processing-widerface/4/output

TF Object Detection code: martinhoest/datasets/tf_models/1


## Run command

```
floyd login
```

```
floyd init '...'
```

### For TrainingFiles_1

```
floyd run --gpu --env tensorflow-1.5 --data zhouji1994/datasets/widerface/1:training --data martinhoest/datasets/tf_models/1:tf_models --tensorboard 'bash run.sh'
```

### For TrainingFiles_2

#### Training

```
floyd run --gpu --env tensorflow-1.5 --data zhouji1994/datasets/widerface/1:training --data martinhoest/datasets/tf_models/1:tf_models --data zhouji1994/projects/processing-widerface/4/output:/data --tensorboard 'bash run.sh'
```

#### Eval

```
floyd run --gpu --env tensorflow-1.5 --data zhouji1994/datasets/widerface/1:training --data martinhoest/datasets/tf_models/1:tf_models --data zhouji1994/projects/face-detection/27/output:model --data zhouji1994/projects/processing-widerface/4/output:/data --tensorboard 'bash run_eval.sh'
```
