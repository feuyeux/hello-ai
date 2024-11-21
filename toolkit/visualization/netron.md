# netron

<https://github.com/lutzroeder/netron>

<https://netron.app>

```sh
pip install netron
```

```python
import netron

netron.start('/mnt/d/park/detectron2/model.onnx')
```

| 序号 | 框架名称       | 对应文件名称                   |
|------|----------------|-------------------------------|
| 1    | ONNX           | .onnx, .pb, .pbtxt            |
| 4    | Keras          | .h5, .keras                  |
| 5    | Core ML       | .mlmodel                     |
| 7    | Caffe         | .caffemodel, .prototxt       |
| 9    | Caffe2        | Predict_net.pb               |
| 11   | Darknet       | .cfg                         |
| 13   | MxNet         | .model, -symbol.json         |
| 16   | Barracuda     | .nn                          |
| 18   | Ncbb          | .param                       |
| 20   | Tengine       | .tmfile                      |
| 22   | TNN           | .tnnproto             |
| 25   | UFF           | .uff                         |
| 27   | Tensorflow Lite | .tflite

| 序号 | 框架名称       | 对应文件名称                   |
|------|----------------|-------------------------------|
| 2    | TorchScript    | .pt, .pth                    |
| 3    | Pytorch        | .pt,.pth                       |
| 3    | Torch        | .t7                          |
| 6    | Arm NN        | .armnn                       |
| 8    | BigDL         | .bigdl, .model               |
| 10   | Chainer       | .npz, .h5                    |
| 12   | CNTK          | .model, .cntk                |
| 14   | Deeplearning4j | .zip                         |
| 17   | MediaPipe     | .pbtxt                       |
| 19   | ML.NET        | .zip                         |
| 21   | MNN           | .mnn                         |
| 23   | PaddlePaddle  | .zip, _model_                |
| 24   | OpenVINO      | .xml                         |
| 26   | Scikit-learn  | .pkl                         |
| 28   | TensorFlowjs   | Model.json, .pb              |
| 29   | Tensorflow    | .pb, .meta, .pbtxt, .ckpt, .index |
