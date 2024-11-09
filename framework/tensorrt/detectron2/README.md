# Detectron 2 Mask R-CNN R50-FPN 3x in TensorRT

<https://github.com/NVIDIA/TensorRT/tree/main/samples/python/detectron2>

Support for Detectron 2 Mask R-CNN R50-FPN 3x model in TensorRT. This script helps with converting, running and validating this model with TensorRT.

## 1 Setup

install pytorch

```sh
conda create --name torch_trt python=3.12
conda activate torch_trt
conda config --append channels conda-forge
conda config --append channels nvidia
# conda install cudatoolkit
conda install -c pytorch torchvision cudatoolkit=11.8
```

```sh
python /mnt/d/coding/hello-ai/frameworks/tensorrt/detectron2/test_torch.py
```

install onnx .. opencv

```sh
pip install onnx==1.16.1 onnxruntime==1.18.1 Pillow>=10.0.0 cuda-python==12.5.0 pyyaml==6.0.1 requests==2.32.2 tqdm==4.66.4 numpy==1.26.4 opencv-python
```

install detectron2

```sh
# https://github.com/facebookresearch/detectron2.git
python -m pip install -e "/mnt/d/coding/detectron2"

Successfully installed detectron2-0.6
```

install onnx_graphsurgeon

```sh
pip install git+https://github.com/NVIDIA/TensorRT#subdirectory=tools/onnx-graphsurgeon --proxy=http://localhost:57885

Successfully installed onnx_graphsurgeon-0.5.2
```

install tensorrt

```sh
pip install tensorrt_cu12_libs==10.2.0 tensorrt_cu12_bindings==10.2.0 tensorrt==10.2.0 --extra-index-url https://pypi.nvidia.com

Successfully installed nvidia-cuda-runtime-cu12-12.5.82 tensorrt-10.2.0 tensorrt-cu12-10.2.0 tensorrt_cu12_bindings-10.2.0 tensorrt_cu12_libs-10.2.0
```

## 2 Model Conversion

workflow: Detectron 2 → ONNX → TensorRT

### Detectron 2 Deployment

```sh
export detectron2_src=/mnt/d/coding/detectron2
export export_model_py=$detectron2_src/tools/deploy/export_model.py
# https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
export mask_rcnn_R_50_FPN_3x_model=/mnt/d/park/model_final_f10217.pkl
export mask_rcnn_R_50_FPN_3x_config="$detectron2_src/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
# https://cdn.shopify.com/s/files/1/0015/5084/3975/products/website-7793_1344x1344.jpg
export image_1344_1344=/mnt/d/coding/hello-ai/frameworks/tensorrt/detectron2/img/1344_1344.jpg
```

```sh
python $export_model_py \
    --sample-image $image_1344_1344 \
    --config-file $mask_rcnn_R_50_FPN_3x_config \
    --export-method tracing \
    --format onnx \
    --output "/mnt/d/park/detectron2" \
    MODEL.WEIGHTS $mask_rcnn_R_50_FPN_3x_model \
    MODEL.DEVICE cuda

[07/29 23:46:25 detectron2]: Success.
```

- `--sample-image` is 1344x1344 image;
- `--config-file` path to Mask R-CNN R50-FPN 3x config, included with detectron2;
- `MODEL.WEIGHTS` are weights of Mask R-CNN R50-FPN 3x that can be downloaded [here](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md).
- Resulted `model.onnx` will be an input to conversion script.

```sh
$ ls /mnt/d/park/detectron2
model.onnx
```

### Create ONNX Graph

```sh
cd /mnt/d/coding/hello-ai/frameworks/tensorrt/detectron2/
python create_onnx.py \
    --exported_onnx /mnt/d/park/detectron2/model.onnx \
    --onnx /mnt/d/park/detectron2/converted.onnx \
    --det2_config $mask_rcnn_R_50_FPN_3x_config \
    --det2_weights $mask_rcnn_R_50_FPN_3x_model \
    --sample_image $image_1344_1344

INFO:ModelHelper:Saved ONNX model to /mnt/d/park/detectron2/converted.onnx
```

- Netron:(tools/netron.md)
- NMS(Non-Maximum Suppression 非极大值抑制)

efficientNMS 一种在目标检测任务中常用的算法，用于去除多余的边界框（bounding boxes），从而只保留最有可能包含目标的边界框。主要步骤：

- 排序(Score Sorting)：首先，根据置信度分数对所有边界框进行排序。置信度分数通常表示边界框中包含目标的可能性。
- 选择(Selection)：选择置信度分数最高的边界框作为当前边界框。
- 抑制(Suppression)：将当前边界框与其他边界框进行比较。如果两个边界框的IoU(Intersection over Union 交并比)超过预设的阈值，则抑制（即删除）其他边界框。
- 迭代(Iteration)：重复上述步骤，直到所有边界框都被处理完毕。

### Build TensorRT Engine

```sh
trtexec --onnx=/mnt/d/park/detectron2/converted.onnx --saveEngine=/mnt/d/park/detectron2/engine.trt --useCudaGraph
```

FP16 Precision

```sh
python3 build_engine.py \
    --onnx /mnt/d/park/detectron2/converted.onnx \
    --engine /mnt/d/park/detectron2/engine_fp16.trt \
    --precision fp16

[07/30/2024-23:51:47] [TRT] [I] Local timing cache in use. Profiling results in this builder pass will not be stored.
[07/30/2024-23:54:02] [TRT] [I] [GraphReduction] The approximate region cut reduction algorithm is called.
[07/30/2024-23:54:02] [TRT] [I] Detected 1 inputs and 5 output network tensors.
[07/30/2024-23:54:05] [TRT] [I] Total Host Persistent Memory: 486512
[07/30/2024-23:54:05] [TRT] [I] Total Device Persistent Memory: 6656
[07/30/2024-23:54:05] [TRT] [I] Total Scratch Memory: 9024512
[07/30/2024-23:54:05] [TRT] [I] [BlockAssignment] Started assigning block shifts. This will take 151 steps to complete.
[07/30/2024-23:54:05] [TRT] [I] [BlockAssignment] Algorithm ShiftNTopDown took 13.0651ms to assign 14 blocks to 151 nodes requiring 282197504 bytes.
[07/30/2024-23:54:05] [TRT] [I] Total Activation Memory: 282196480
[07/30/2024-23:54:05] [TRT] [I] Total Weights Memory: 92405968
[07/30/2024-23:54:05] [TRT] [I] Engine generation completed in 137.912 seconds.
[07/30/2024-23:54:05] [TRT] [I] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 24 MiB, GPU 444 MiB
[07/30/2024-23:54:05] [TRT] [I] [MemUsageStats] Peak memory usage during Engine building and serialization: CPU: 4577 MiB
INFO:EngineBuilder:Serializing engine to file: /mnt/d/park/detectron2/engine_fp16.trt
```

INT8 Precision

```sh
# http://images.cocodataset.org/zips/train2017.zip
export calib_input=/mnt/d/park/train2017
python3 build_engine.py \
    --onnx /mnt/d/park/detectron2/converted.onnx \
    --engine /mnt/d/park/detectron2/engine_int8.trt \
    --precision int8 \
    --calib_input $calib_input \
    --calib_cache /mnt/d/park/detectron2/calibration.cache

[07/31/2024-01:19:08] [TRT] [I] Local timing cache in use. Profiling results in this builder pass will not be stored.
[07/31/2024-01:23:03] [TRT] [I] [GraphReduction] The approximate region cut reduction algorithm is called.
[07/31/2024-01:23:03] [TRT] [I] Detected 1 inputs and 5 output network tensors.
[07/31/2024-01:23:08] [TRT] [I] Total Host Persistent Memory: 400496
[07/31/2024-01:23:08] [TRT] [I] Total Device Persistent Memory: 26112
[07/31/2024-01:23:08] [TRT] [I] Total Scratch Memory: 9024512
[07/31/2024-01:23:08] [TRT] [I] [BlockAssignment] Started assigning block shifts. This will take 140 steps to complete.
[07/31/2024-01:23:08] [TRT] [I] [BlockAssignment] Algorithm ShiftNTopDown took 11.3956ms to assign 13 blocks to 140 nodes requiring 241872384 bytes.
[07/31/2024-01:23:08] [TRT] [I] Total Activation Memory: 241871872
[07/31/2024-01:23:08] [TRT] [I] Total Weights Memory: 50546464
[07/31/2024-01:23:09] [TRT] [I] Engine generation completed in 240.851 seconds.
[07/31/2024-01:23:09] [TRT] [I] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 16 MiB, GPU 667 MiB
[07/31/2024-01:23:09] [TRT] [I] [MemUsageStats] Peak memory usage during Engine building and serialization: CPU: 4423 MiB
INFO:EngineBuilder:Serializing engine to file: /mnt/d/park/detectron2/engine_int8.trt
```

## 3 Inference

```sh
python infer.py \
    --engine /mnt/d/park/detectron2/engine.trt \
    --input /mnt/d/park/images \
    --det2_config $mask_rcnn_R_50_FPN_3x_config \
    --output /mnt/d/park/images_outputs
```

valutions

```sh
# http://images.cocodataset.org/zips/val2017.zip
export val_input=/mnt/d/park/val2017
# https://huggingface.co/datasets/merve/coco/tree/main/annotations
export DETECTRON2_DATASETS=/mnt/d/park/datasets
python eval_coco.py \
    --engine /mnt/d/park/detectron2/engine_int8.trt \
    --input $val_input \
    --det2_config $mask_rcnn_R_50_FPN_3x_config \
    --det2_weights $mask_rcnn_R_50_FPN_3x_model

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.364
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.579
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.390
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.181
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.388
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.514
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.303
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.467
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.489
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.310
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.519
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.630
```

## 4 netron

```sh
pip install netron
```

```python
import netron

netron.start('/mnt/d/park/detectron2/model.onnx')
```
