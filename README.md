# dagnn_caffe_deploy
MatConvNet-DAGNN to Caffe converter

Initial **EXPERIMENTAL** version of a converter from a MatConvNet DagNN model to Caffe.

Only a limited subset of layer types is currently supported: dagnn.Conv, dagnn.ReLU, dagnn.Concat, dagnn.BatchNorm, dagnn.Sum,     dagnn.Pooling, dagnn.LRN, dagnn.SoftMax, dagnn.Loss, dagnn.DropOut

Sucessfully tested with:

- [imagenet-resnet-50-dag](http://www.vlfeat.org/matconvnet/models/imagenet-resnet-50-dag.mat)
- [imagenet-matconvnet-alex](http://www.vlfeat.org/matconvnet/models/imagenet-matconvnet-alex.mat)
- [imagenet-googlenet-dag](http://www.vlfeat.org/matconvnet/models/imagenet-googlenet-dag.mat)

Classification tests with the converted version of imagenet-resnet-50-dag are the same than with the [MSRA version of Resnet-50 caffe model](https://github.com/KaimingHe/deep-residual-networks).

Using:

- [Caffe rc-3](https://github.com/BVLC/caffe/releases/tag/rc3)
- [MatConvNet 1.0-beta23](https://github.com/vlfeat/matconvnet/releases/tag/v1.0-beta23)

## Installation

Just copy the files in this repository within the MatConvNet folder.

Before using dagnn_caffe_deploy, remember to add MatConvNet's "matlab" folder to the path. MatCaffe must also be available.

## Sample Usage

```
net = load('imagenet-resnet-50-dag.mat');          % Load the MatConvNet model
dagnn_caffe_deploy(net,'imagenet-resnet-50-dag');  % Run the converter speciiying the model
                                                   % and the prefix for the output files
```

The instructions above will produce the following files:

- imagenet-resnet-50-dag.caffemodel
- imagenet-resnet-50-dag.prototxt
- imagenet-resnet-50-dag_mean_image.binaryproto
