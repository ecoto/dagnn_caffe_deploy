function tnet = dagnn_tidy(net)
%DAGNN_TIDY  Fix an incomplete or outdated dagnn network
%   NET = DAGNN_TIDY(NET) takes the NET object and upgrades
%   it to the current version of MatConvNet. This is necessary in
%   order to allow MatConvNet to evolve, while maintaining the NET
%   objects clean. This function ignores custom layers.
%
%   The function is also generally useful to fill in missing default
%   values in NET.
%
%   Based on: VL_SIMPLENN_TIDY().
%
% Copyright (C) 2017 Ernesto Coto
%                    Visual Geometry Group, University of Oxford.
% All rights reserved.
%
% This file is made available under the terms of the BSD license.


tnet = struct('layers', {{}}, 'params', {{}}, 'meta', struct()) ;

% copy meta information in net.meta subfield
if isfield(net, 'meta')
  tnet.meta = net.meta ;
end

if isfield(net, 'classes')
  tnet.meta.classes = net.classes ;
end

if isfield(net, 'normalization')
  tnet.meta.normalization = net.normalization ;
end

% copy params
for l = 1:numel(net.params)
  param = net.params(l) ;
  % save back
  tnet.params{l} = param ;
end

% check weights format
for l = 1:numel(net.layers)
  defaults = {};
  layer = net.layers(l) ;

  % check weights format
  switch layer.type
    case {'dagnn.Conv', 'dagnn.ConvTranspose', 'dagnn.BatchNorm'}
      if ~isfield(layer, 'weights')
          layer.weights = {};
          for bn_i=1:numel(layer.params)
            % save values of all parameters

            % TO CHECK: Always copy all parameters or restrict to filters,
            % biases and moments ??
            %if ~isempty(strfind(layer.params{bn_i}, 'filter')) || ...
            %   ~isempty(strfind(layer.params{bn_i}, 'bias')) || ...
            %   ~isempty(strfind(layer.params{bn_i}, 'moments'))
                param_name = layer.params{bn_i};
                param_layer = tnet.params(cellfun(@(l) strcmp(l.name, param_name), tnet.params));
                values = param_layer{1}.value;
                layer.weights = [ layer.weights,  {values}];
            %end
          end
      end
  end
  if ~isfield(layer, 'weights')
    layer.weights = {} ;
  end

  % Check that weights include moments in batch normalization.
  if strcmp(layer.type, 'dagnn.BatchNorm')
   if numel(layer.weights) < 3
     layer.weights{3} = ....
       zeros(numel(layer.weights),2,'single') ;
   end
  end

  % Fill in missing values.
  switch layer.type
    case 'dagnn.Conv'
      defaults = [ defaults {...
        'pad', 0, ...
        'stride', 1, ...
        'dilate', 1, ...
        'opts', {}}] ;

    case 'dagnn.Pooling'
      defaults = [ defaults {...
        'pad', 0, ...
        'stride', 1, ...
        'opts', {}}] ;

    case 'dagnn.ConvTranspose'
      defaults = [ defaults {...
        'crop', 0, ...
        'upsample', 1, ...
        'numGroups', 1, ...
        'opts', {}}] ;

% TO CHECK: Is this case really intentionally duplicated?
%     case {'dagnn.Pooling'}
%       defaults = [ defaults {...
%         'method', 'max', ...
%         'pad', 0, ...
%         'stride', 1, ...
%         'opts', {}}] ;

    case 'dagnn.ReLU'
      defaults = [ defaults {...
        'leak', 0}] ;

    case 'dagnn.DropOut'
      defaults = [ defaults {...
        'rate', 0.5}] ;

    case 'dagnn.LRN'
      defaults = [ defaults {...
        'param', [5 1 0.0001/5 0.75]}] ;

% TO CHECK: what is the equivalent of pdist in DagNN?
%     case {'pdist'}
%       defaults = [ defaults {...
%         'noRoot', false, ...
%         'aggregate', false, ...
%         'p', 2, ...
%         'epsilon', 1e-3, ...
%         'instanceWeights', []} ];

    case 'dagnn.BatchNorm'
      defaults = [ defaults {...
        'epsilon', 1e-5 } ] ;
  end

  for i = 1:2:numel(defaults)
    if ~isfield(layer.block, defaults{i})
      layer.(defaults{i}) = defaults{i+1} ;
    end
  end

   % save back
  tnet.layers{l} = layer ;
end
