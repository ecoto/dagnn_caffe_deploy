function dagnn_caffe_deploy(net, caffeModelBaseName, varargin)
% DAGNN_CAFFE_DEPLOY export a DagNN network to Caffe model
%   DAGNN_CAFFE_DEPLOY(NET, CAFFE_BASE_MODELNAME)
%   Export a dagnn network NET to a Caffe model.
%   The caffe model is stored in the following files:
%
%     [CAFFE_BASE_MODELNAME '.prototxt']   - Network definition file
%     [CAFFE_BASE_MODELNAME '.caffemodel'] - Binary Caffe model
%     [CAFFE_BASE_MODELNAME '_mean_image.binaryproto'] (optional) -
%         The average image (if set in net.normalization.averageImage)
%
%   Compiled MatCaffe (usually located in `<caffe_dir>/matlab`, built
%   with the `matcaffe` target) must be in path.
%
%   Only a limited subset of layers is currently supported and those are:
%
%     dagnn.Conv, dagnn.ReLU, dagnn.Concat, dagnn.BatchNorm, dagnn.Sum,
%     dagnn.Pooling, dagnn.LRN, dagnn.SoftMax, dagnn.Loss, dagnn.DropOut
%
%   Please note that thanks to different implementations, the outputs of
%   dagnn and Caffe models are not neccessarily identical.
%
%   DAGNN_CAFFE_DEPLOY(NET, CAFFE_BASE_MODELNAME, 'OPT', VAL, ...)
%   takes the following options:
%
%   `removeDropout`:: `true`
%      When true, do not deploy dropout layers.
%
%   `replaceSoftMaxLoss`:: `true`
%      Replace SoftMax log loss with SoftMax.
%
%   `inputBlobName`:: 'data'
%      Name of the input data blob in the final protobuf.
%
%   `labelBlobName`:: 'label'
%      Name of the input label blob in the final protobuf.
%
%   `outputBlobName`:: 'prob'
%      Name of the output blob in the resulting protobuf.
%
%   `silent`:: false
%      When true, suppresses all output to stdout.
%
%  Based on: SIMPLENN_CAFFE_DEPLOY()
%  See Also: DAGNN_TIDY()
%
% Copyright (C) 2017 Ernesto Coto, Samuel Albanie.
%                    Visual Geometry Group, University of Oxford.
% All rights reserved.
%
% This file is made available under the terms of the BSD license.

opts.inputBlobName = 'data';
opts.outputBlobName = 'prob';
opts.labelBlobName = 'label';
opts.removeDropout = true;
opts.replaceSoftMaxLoss = true;

% TODO: Not able to test yet
%opts.doTest = true;
%opts.testData = [];

opts.silent = false;
opts = vl_argparse(opts, varargin);
if ~exist('caffe.Net', 'class'), error('MatCaffe not in path.'); end

info = @(varargin) fprintf(1, varargin{:});
if opts.silent, info = @(varargin) []; end;

info('Exporting dagnn model to caffe model %s\n', caffeModelBaseName);
[modelDir, name] = fileparts(caffeModelBaseName);
[~,~,~] = mkdir(modelDir);

% -------------------------------------------------------------------------
%                                                          Tidy the network
% -------------------------------------------------------------------------

% load into DagNN object
dagnn_net = dagnn.DagNN.loadobj(net);

% move to cpu
dagnn_net.move('cpu');

% convert to a vanilla MATLAB structure
net = dagnn_net.saveobj();

% clean up
net = dagnn_tidy(net);

% Remove dropout layers
if opts.removeDropout
  net.layers(cellfun(@(l) strcmp(l.type, 'dagnn.DropOut'), net.layers)) = [];
end

if opts.replaceSoftMaxLoss
  % If last layer is softmax loss, replace it with softmax
  ll = net.layers{end};
  if strcmp(ll.type, 'dagnn.Loss') || ...
      (strcmp(ll.type, 'dagnn.Loss') && strcmp(ll.loss, 'softmaxlog'))
    net.layers{end}.type = 'dagnn.SoftMax';
  elseif isequal(net.layers{end}.type, 'dagnn.Loss')
    error('Unsupported loss function: %s', net.layers{end}.loss);
  end
end

% TO CHECK: Is this really needed ?.
%for idx = 1:numel(net.layers)
%  % Add missing layer names
%  if ~isfield(net.layers{idx}, 'name')
%    net.layers{idx}.name = sprintf('layer%d', idx);
%  end
%end

avImage = [];
if isfield(net.meta, 'normalization') && ...
    isfield(net.meta.normalization, 'imageSize')
  imSize = net.meta.normalization.imageSize;
  if isfield(net.meta.normalization, 'averageImage')
    avImage = net.meta.normalization.averageImage;
    if numel(avImage) == imSize(3)
      avImage = reshape(avImage, 1, 1, imSize(3));
    end
  end
else
  error('Missing image size. Please set `net.normalization.imageSize`.');
end

% -------------------------------------------------------------------------
%                                                           Export prototxt
% -------------------------------------------------------------------------
prototxtFilename = [caffeModelBaseName '.prototxt'];
fid = fopen(prototxtFilename, 'w');

fprintf(fid, 'name: "%s"\n\n', name); % Network name

% Export input dimensions
fprintf(fid, 'input: "%s"\n', opts.inputBlobName);
fprintf(fid, 'input_dim: 1\n');
fprintf(fid, 'input_dim: %d\n', imSize(3));
fprintf(fid, 'input_dim: %d\n', imSize(1));
fprintf(fid, 'input_dim: %d\n\n', imSize(2));

% Use this to keep track of data input sizes at each layer
varSizes =  dagnn_net.getVarSizes({opts.inputBlobName, imSize});

isFullyConnected = false(size(net.layers));
for idx = 1:numel(net.layers)
  % create layer entry in prototxt
  fprintf(fid,'layer {\n');
  fprintf(fid,'  name: "%s"\n', net.layers{idx}.name); % Layer name
  switch net.layers{idx}.type
    case 'dagnn.Conv'
      % Find this layer's input variable
      % TODO: Find a faster way to do this ?
      inputSizeIndex = 0;
      input_name = net.layers{idx}.inputs{1}; % TO CHECK: The first one should suffice ?
      while inputSizeIndex < numel(varSizes)
         inputSizeIndex = inputSizeIndex + 1;
         if strcmp(dagnn_net.vars(inputSizeIndex).name, input_name)
             break;
         end
      end
      % Get the layer size to be able to differentiate between a Conv layer
      % and a FullyConnected layer. Also needed to compute groups.
      layerInputSize = varSizes{inputSizeIndex};
      if numel(layerInputSize) == 2
        layerInputSize(3) = 1;
      end
      filtH = net.layers{idx}.block.size(1);
      filtW = net.layers{idx}.block.size(2);
      if filtH < layerInputSize(1) || filtW < layerInputSize(2)
        % Convolution layer
        fprintf(fid, '  type: "Convolution"\n');
        write_connection(fid, net.layers, idx);
        fprintf(fid, '  convolution_param {\n');
        write_kernel(fid, [filtH, filtW]);
        fprintf(fid, '    num_output: %d\n', net.layers{idx}.block.size(4));
        write_stride(fid,net.layers{idx}.block.stride);
        if isfield(net.layers{idx}.block, 'pad') && numel(net.layers{idx}.block.pad) == 4
          % Make sure pad is symmetrical
          if any(net.layers{idx}.block.pad([1, 3]) ~= net.layers{idx}.block.pad([2, 3]))
            error('Caffe only supports symmetrical padding');
          end
        end
        write_pad(fid, net.layers{idx}.block.pad);
        numGroups = layerInputSize(3) / size(net.layers{idx}.weights{1}, 3);
        assert(mod(numGroups, 1) == 0);
        if numGroups > 1
          fprintf(fid, '    group: %d\n', numGroups);
        end
        fprintf(fid, '  }\n');
      elseif filtH == layerInputSize(1) && filtW == layerInputSize(2)
         % Fully connected layer
         isFullyConnected(idx) = true;
         fprintf(fid, '  type: "InnerProduct"\n');
         write_connection(fid, net.layers, idx);
         fprintf(fid, '  inner_product_param {\n');
         fprintf(fid, '    num_output: %d\n', net.layers{idx}.block.size(4));
         fprintf(fid, '  }\n');
      else
        error('Filter size (%d,%d) is larger than input size (%d,%d)', ...
          filtH, filtW, layerInputSize(1), layerInputSize(2))
      end

    case 'dagnn.ReLU'
      fprintf(fid, '  type: "ReLU"\n');
      write_connection(fid, net.layers, idx);

      case 'dagnn.Concat'
       mcnDim =  net.layers{idx}.block.dim ;
       caffeMcnMap = [2 3 1 0] ;
       caffeAxis = caffeMcnMap(mcnDim) ;
       fprintf(fid, '  type: "Concat"\n');
       write_connection(fid, net.layers, idx);
       fprintf(fid, '  concat_param {\n');
       fprintf(fid, '    axis: %d\n', caffeAxis);
       fprintf(fid, '  }\n');

     case 'dagnn.BatchNorm'
       fprintf(fid, '  type: "BatchNorm"\n');
       write_connection(fid, net.layers, idx);
       fprintf(fid, '  batch_norm_param {\n');
       % indicate caffe to use the stored mean/variance estimates
       fprintf(fid, '    use_global_stats: true\n');
       fprintf(fid, '    eps: %f\n', net.layers{idx}.block.epsilon);
       fprintf(fid, '  }\n');

       % Add extra Scale layer
       fprintf(fid,'}\n\n');
       fprintf(fid,'layer {\n');
       % Change layer name
       scale_layer_name = strrep(net.layers{idx}.name, 'bn', 'scale');
       fprintf(fid,'  name: "%s"\n', scale_layer_name);
       fprintf(fid, '  type: "Scale"\n');
       fprintf(fid, '  bottom: "%s"\n', net.layers{idx}.name);
       fprintf(fid, '  top: "%s"\n', scale_layer_name);
       fprintf(fid, '  scale_param {\n');
       % indicate caffe there is a bias parameter, since in
       % theory, there is always one, even if it's a vector of zeros
       fprintf(fid, '    bias_term: true\n');
       fprintf(fid, '  }\n');

    case 'dagnn.Sum'
      fprintf(fid, '  type: "Eltwise"\n');
      write_connection(fid, net.layers, idx);

    case 'dagnn.Pooling'
      fprintf(fid, '  type: "Pooling"\n');
      % Check padding compatibility with caffe. See:
      % http://www.vlfeat.org/matconvnet/matconvnet-manual.pdf
      % for more details.
      if numel(net.layers{idx}.block.pad) == 1
        net.layers{idx}.block.pad = repmat(net.layers{idx}.block.pad, 1, 4);
      end
      if numel(net.layers{idx}.block.stride) == 1
        net.layers{idx}.block.stride = repmat(net.layers{idx}.block.stride, 1, 2);
      end
      if numel(net.layers{idx}.block.poolSize) == 1
        net.layers{idx}.block.pool = repmat(net.layers{idx}.block.poolSize, 1, 2);
      end
      pad = net.layers{idx}.block.pad;
      if pad([2, 4]) == net.layers{idx}.block.poolSize - 1
        pad([2, 4]) = 0;
      else
        pad([2, 4]) = pad([2, 4]) - net.layers{idx}.block.stride + 1;
      end
      % Some older versions did not use these upper bounds
      pad = max(pad, 0);

      write_connection(fid, net.layers, idx);
      fprintf(fid, '  pooling_param {\n');
      switch (net.layers{idx}.block.method)
        case 'max'
          caffe_pool = 'MAX';
        case 'avg'
          caffe_pool = 'AVE';
        otherwise
          error('Unknown pooling type');
      end
      fprintf(fid, '    pool: %s\n', caffe_pool);
      write_kernel(fid, net.layers{idx}.block.poolSize);
      write_stride(fid, net.layers{idx}.block.stride);
      write_pad(fid, pad);
      fprintf(fid, '  }\n');

    case 'dagnn.LRN'
      % MATLAB param = [local_size, kappa, alpha/local_size, beta]
      fprintf(fid, '  type: "LRN"\n');
      write_connection(fid, net.layers, idx);
      fprintf(fid, '  lrn_param {\n');
      fprintf(fid, '    local_size: %d\n', net.layers{idx}.block.param(1));
      fprintf(fid, '    k: %f\n', net.layers{idx}.block.param(2));
      fprintf(fid, '    alpha: %f\n', net.layers{idx}.block.param(3)*net.layers{idx}.block.param(1));
      fprintf(fid, '    beta: %f\n', net.layers{idx}.block.param(4));
      fprintf(fid, '  }\n');

    case 'dagnn.SoftMax'
      fprintf(fid, '  type: "Softmax"\n');
      write_connection(fid, net.layers, idx);

    case 'dagnn.Loss'
      fprintf(fid, '  type: "SoftmaxWithLoss"\n');
      write_connection(fid, net.layers, idx, true);

    % TODO: Find a network where to test this !
    case 'dagnn.DropOut'
      fprintf(fid, '  type: "Dropout"\n');
      write_connection(fid, net.layers, idx);
      fprintf(fid, '  dropout_param {\n');
      fprintf(fid, '    dropout_ratio: %d\n', net.layers{idx}.block.rate);
      fprintf(fid, '  }\n');

    otherwise
      error('Unknown layer type: %s', net.layers{idx}.type);
  end
  fprintf(fid,'}\n\n');
end
fclose(fid);
info('Network definition exported to: %s.\n', prototxtFilename);

% -------------------------------------------------------------------------
%                                                         Export caffemodel
% -------------------------------------------------------------------------
caffe.set_mode_cpu();
caffeNet = caffe.Net(prototxtFilename, 'test');
firstConv = true;
for idx = 1:numel(net.layers)
  layer_type = net.layers{idx}.type;
  layer_name = net.layers{idx}.name;
  switch layer_type
    case 'dagnn.Conv'
      filters = net.layers{idx}.weights{1};
      % Convert from HxWxCxN to WxHxCxN per Caffe's convention
      filters = permute(filters, [2 1 3 4]);
      if firstConv
        if size(filters, 3) == 3
          % We assume this is RGB Conv., need to convert RGB to BGR
          filters = filters(:,:, [3 2 1], :);
        end
        firstConv = false; % Do this only for first convolution;
      end
      if isFullyConnected(idx)
        % Fully connected layer, squeeze to 2 dims
        filters = reshape(filters, [], size(filters, 4));
      end
      caffeNet.layers(layer_name).params(1).set_data(filters); % set weights
      hasBias = numel(net.layers{idx}.params)>1 ;
      % If there is a bias parameter
      if hasBias
          biases = net.layers{idx}.weights{2}(:);
          caffeNet.layers(layer_name).params(2).set_data(biases); % set bias
      end
    case 'dagnn.BatchNorm'
        moments = net.layers{idx}.weights{3}; % first two are for scaling, third one should correspond to the moments
        mean = moments(:,1);
        caffeNet.layers(layer_name).params(1).set_data(mean); % set mean
        variance_plus_eps = moments(:,2).^2;
        variance = variance_plus_eps - net.layers{idx}.block.epsilon;
        caffeNet.layers(layer_name).params(2).set_data(variance); % set variance
        scale_factor = 1; % assume scale factor always one, since it can't really be calculated
        caffeNet.layers(layer_name).params(3).set_data(scale_factor); % set scale factor

        % Add parameters for extra Scale layer
        scale_layer_name = strrep(net.layers{idx}.name, 'bn', 'scale');
        mult = net.layers{idx}.weights{1};
        caffeNet.layers(scale_layer_name).params(1).set_data(mult); % set mult
        bias = net.layers{idx}.weights{2};
        caffeNet.layers(scale_layer_name).params(2).set_data(bias); % set bias

    case {'dagnn.ReLU', 'dagnn.LRN', 'dagnn.Pooling' , 'dagnn.SoftMax', 'dagnn.Sum', 'dagnn.Concat' }
      % No weights - nothing to do
    otherwise
      error('Unknown layer type %s', layer_type)
  end
end
modelFilename = [caffeModelBaseName '.caffemodel'];
caffeNet.save(modelFilename);
delete(caffeNet);
info('Model file exported to: %s.\n', modelFilename);

% -------------------------------------------------------------------------
%                                                        Export mean image
% -------------------------------------------------------------------------
if ~isempty(avImage)
  if size(avImage, 1) == 1 && size(avImage, 2) == 1
    % Single value, we'll duplicate it to im_size
    avImage = repmat(avImage, imSize(1), imSize(2));
  end
  avImage = matlab_img_to_caffe(avImage);
  meanFilename = [caffeModelBaseName, '_mean_image.binaryproto'];
  caffe.io.write_mean(avImage, meanFilename)
  info('Mean image exported to: %s.\n', meanFilename);
end

% TODO: Not able to test yet
%if opts.doTest
%  simplenn_caffe_compare(net, caffeModelBaseName, opts.testData);
%end

  function write_stride(fid, stride)
      if numel(stride) == 1
        fprintf(fid, '    stride: %d\n', stride);
      elseif numel(stride) == 2
        fprintf(fid, '    stride_h: %d\n', stride(1));
        fprintf(fid, '    stride_w: %d\n', stride(2));
      end
  end

  function write_kernel(fid, kernelSize)
    if numel(kernelSize) == 1
      fprintf(fid, '    kernel_size: %d\n', kernelSize);
    elseif numel(kernelSize) == 2
      fprintf(fid, '    kernel_h: %d\n', kernelSize(1));
      fprintf(fid, '    kernel_w: %d\n', kernelSize(2));
    end
  end


  function write_pad(fid, pad)
    if numel(pad) == 1
      fprintf(fid, '    pad: %d\n', pad);
    elseif numel(pad) == 4
      fprintf(fid, '    pad_h: %d\n', pad(1));
      fprintf(fid, '    pad_w: %d\n', pad(2));
    else
      error('pad vector size must be 1 or 4')
    end
  end

  function write_connection(fid, layers, idx, isLoss)
    if idx == 1
      bottom_name = opts.inputBlobName;
      fprintf(fid, '  bottom: "%s"\n', bottom_name);
    else
      for bottom_inputs_idx = 1:numel(layers{idx}.inputs)
          input_name = layers{idx}.inputs{bottom_inputs_idx};
          for input_search_idx=idx-1:-1:1
            bottom_layer = layers(cellfun(@(l) strcmp(l, input_name), layers{input_search_idx}.outputs));
            if ~isempty(bottom_layer)
                if strcmp(layers{input_search_idx}.type,'dagnn.BatchNorm')
                    % in this case, a Scale layer should have been added
                    % rigth after the BatchNorm layer, so make the
                    % Scale layer the bottom of this one
                    scale_layer_name = strrep(layers{input_search_idx}.name, 'bn', 'scale');
                    fprintf(fid, '  bottom: "%s"\n', scale_layer_name);
                else
                    fprintf(fid, '  bottom: "%s"\n', layers{input_search_idx}.name);
                end
                break;
            end
          end
      end
    end
    top_name = layers{idx}.name;
    if idx == numel(layers) && ~isempty(opts.outputBlobName)
      top_name = opts.outputBlobName;
    end
    if nargin > 3 && isLoss
      fprintf(fid, '  bottom: "%s"\n', opts.labelBlobName);
    end
    fprintf(fid, '  top: "%s"\n', top_name);
  end


  function img = matlab_img_to_caffe(img)
    img = single(img);
    % Convert from HxWxCxN to WxHxCxN per Caffe's convention
    img = permute(img, [2 1 3 4]);
    if size(img,3) == 3
      % Convert from RGB to BGR channel order per Caffe's convention
      img = img(:,:, [3 2 1], :);
    end
  end
end
