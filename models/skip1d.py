import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class Concat1d(nn.Module):
    def __init__(self, dim, *args):
        super(Concat1d, self).__init__()
        self.dim = dim

        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, input):
        inputs = []
        for module in self._modules.values():
            try:
                inputs.append(module(input))
            except ValueError:
                pdb.set_trace()

        inputs_shapes2 = [x.shape[2] for x in inputs]        

        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)

            inputs_ = []
            for inp in inputs: 
                diff2 = (inp.size(2) - target_shape2) // 2 
                inputs_.append(inp[:, :, diff2: diff2 + target_shape2])

        return torch.cat(inputs_, dim=self.dim)

    def __len__(self):
        return len(self._modules)

def conv1d(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride'):
    downsampler = None
    if stride != 1 and downsample_mode != 'stride':

        if downsample_mode == 'avg':
            downsampler = nn.AvgPool1d(stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool1d(stride)
        else:
            assert False

        stride = 1

    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad1d(to_pad)
        to_pad = 0
  
    convolver = nn.Conv1d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)

    layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
    return nn.Sequential(*layers)

# class ConvHole1D(nn.Conv1d):
#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         kernel_size,
#         stride=1,
#         dilation=1,
#         bias=True,
#         kernel_initializer=None,
#         device=None,
#         dtype=None
#     ):
#         assert kernel_size % 2 == 1, "Only odd kernels supported for blind spot masking"
#         self.k = kernel_size
#         self.center_idx = kernel_size // 2

#         _w = torch.empty(
#             out_channels,
#             in_channels,
#             kernel_size - 1,
#             device=device,
#             dtype=dtype
#         )

#         if kernel_initializer is None:
#             nn.init.kaiming_uniform_(_w, a=math.sqrt(5))
#         else:
#             nn.init.ones_(_w)

#         super().__init__(
#             in_channels,
#             out_channels,
#             kernel_size,
#             stride=stride,           # now actual stride is supported
#             padding=0,               # no padding here
#             dilation=dilation,
#             bias=bias
#         )

#         self.register_parameter('weight_full', None)
#         self.weight_param = nn.Parameter(_w, requires_grad=True)
#         self.center_weight = nn.Parameter(torch.zeros([out_channels, in_channels, 1]), requires_grad=False)

#     def _assemble_weight(self):
#         return torch.cat([
#             self.weight_param[:, :, :self.center_idx],
#             self.center_weight,
#             self.weight_param[:, :, self.center_idx:]
#         ], dim=2)

#     def forward(self, x):
#         w = self._assemble_weight()
#         return F.conv1d(x, w, self.bias, stride=self.stride, dilation=self.dilation)

class ConvHole1D(nn.Conv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=True,
        kernel_initializer=None,
        device=None,
        dtype=None,
    ):
        assert kernel_size % 2 == 1, "Only odd kernels supported for blind spot masking"
        self.k = kernel_size
        self.dilation = dilation
        self.center_idx = kernel_size // 2

        _w = torch.empty(
            out_channels,
            in_channels,
            kernel_size - 1,
            device=device,
            dtype=dtype
        )

        if kernel_initializer is None:
            nn.init.kaiming_uniform_(_w, a=math.sqrt(5))
        else:
            nn.init.ones_(_w)

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,  # No padding here
            dilation=dilation,
            bias=bias
        )

        self.register_parameter('weight_full', None)
        self.weight_param = nn.Parameter(_w, requires_grad=True)
        self.center_weight = nn.Parameter(torch.zeros([out_channels, in_channels, 1]), requires_grad=False)

    def _assemble_weight(self):
        return torch.cat([
            self.weight_param[:, :, :self.center_idx],
            self.center_weight,
            self.weight_param[:, :, self.center_idx:]
        ], dim=2)

    def forward(self, x):
        # Calculate padding based on dilation
        dilation = self.dilation[0] if isinstance(self.dilation, tuple) else self.dilation
        pad_amt = dilation * (self.k // 2)
        x = F.pad(x, (pad_amt, pad_amt), mode='reflect')  # or 'constant' for zero padding
        w = self._assemble_weight()
        return F.conv1d(x, w, self.bias, stride=self.stride, dilation=self.dilation)

# def convhole1d(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride'):
#     downsampler = None
#     if stride != 1 and downsample_mode != 'stride':
#         if downsample_mode == 'avg':
#             downsampler = nn.AvgPool1d(stride)
#         elif downsample_mode == 'max':
#             downsampler = nn.MaxPool1d(stride)
#         else:
#             raise ValueError(f"Unsupported downsample_mode: {downsample_mode}")
#         stride = 1  # do not use stride in conv, since pool will downsample

#     padder = None
#     to_pad = int((kernel_size - 1) / 2)

#     if pad == 'reflection':
#         padder = nn.ReflectionPad1d(to_pad)
#         pad_mode = 'zero'  # no padding inside ConvHole1D
#     else:
#         pad_mode = pad

#     conv_layer = ConvHole1D(
#         in_channels=in_f,
#         out_channels=out_f,
#         kernel_size=kernel_size,
#         stride=stride,
#         dilation=1,
#         pad_mode=pad_mode,
#         downsample_mode='none',  # handled externally
#         bias=bias
#     )

#     layers = filter(lambda x: x is not None, [padder, conv_layer, downsampler])
#     return nn.Sequential(*layers)

def bn1d(num_features):
    return nn.BatchNorm1d(num_features)

def convhole1d(in_f, out_f, kernel_size, stride=1, bias=True, dilation=1):
    return ConvHole1D(
        in_channels=in_f,
        out_channels=out_f,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        bias=bias
    )

class Swish(nn.Module):
    """
        https://arxiv.org/abs/1710.05941
        The hype was so huge that I could not help but try it
    """
    def __init__(self):
        super(Swish, self).__init__()
        self.s = nn.Sigmoid()

    def forward(self, x):
        return x * self.s(x)

class Sine(nn.Module):
    '''
        Sinusoidal activation layer
    '''
    def __init__(self):
        super(Sine, self).__init__()
        self.omega_0 = 30
    
    def forward(self, x):
        return torch.sin(self.omega_0 * x)

def act(act_fun = 'LeakyReLU'):
    '''
        Either string defining an activation function or module (e.g. nn.ReLU)
    '''
    if isinstance(act_fun, str):
        if act_fun == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=False)
        elif act_fun == 'Swish':
            return Swish()
        elif act_fun == 'ELU':
            return nn.ELU()
        elif act_fun == 'none':
            return nn.Sequential()
        elif act_fun == 'sine':
            return Sine()
        else:
            assert False
    else:
        return act_fun()

#### conv2d ####
class Concat(nn.Module):
    def __init__(self, dim, *args):
        super(Concat, self).__init__()
        self.dim = dim

        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, input):
        inputs = []
        for module in self._modules.values():
            inputs.append(module(input))

        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]        

        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(np.array(inputs_shapes3) == min(inputs_shapes3)):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)

            inputs_ = []
            for inp in inputs: 
                diff2 = (inp.size(2) - target_shape2) // 2 
                diff3 = (inp.size(3) - target_shape3) // 2 
                inputs_.append(inp[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3])

        return torch.cat(inputs_, dim=self.dim)

    def __len__(self):
        return len(self._modules)

def bn(num_features):
    return nn.BatchNorm2d(num_features)
    
def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride'):
    downsampler = None
    if stride != 1 and downsample_mode != 'stride':

        if downsample_mode == 'avg':
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool2d(stride, stride)
        elif downsample_mode  in ['lanczos2', 'lanczos3']:
            downsampler = Downsampler(n_planes=out_f, factor=stride, kernel_type=downsample_mode, phase=0.5, preserve_size=True)
        else:
            assert False

        stride = 1

    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0
  
    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)


    layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
    return nn.Sequential(*layers)

def get_2d_posencode_inp(H, W, n_inputs):
    '''
        Get positionally encoded inputs for inpainting tasks
        
        https://bmild.github.io/fourfeat/
    '''
    X, Y = np.mgrid[:H, :W]
    coords = np.hstack(((10*X/H).reshape(-1, 1), (10*Y/W).reshape(-1, 1)))
    
    freqs = np.random.randn(2, n_inputs)
    
    angles = coords.dot(freqs)
    
    sin_vals = np.sin(2*np.pi*angles)
    cos_vals = np.cos(2*np.pi*angles)
    
    posencode_vals = np.hstack((sin_vals, cos_vals)).astype(np.float32)
    
    inp = posencode_vals.reshape(H, W, 2*n_inputs)

    inp = torch.nn.Parameter(torch.tensor(inp).permute(2, 0, 1)[None, ...])
    
    return inp

def get_inp(tensize, const=10.0):
    '''
        Wrapper to get a variable on graph
    '''
    inp = torch.rand(tensize) /const
    inp = torch.nn.Parameter(inp, requires_grad=True)
    
    return inp

def skip(
        num_input_channels=2, num_output_channels=3, 
        num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], num_channels_skip=[4, 4, 4, 4, 4], 
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        need_sigmoid=True, need_bias=True, 
        pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU', 
        need1x1_up=True):
    """Assembles encoder-decoder with skip connections.

    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

    """
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

    n_scales = len(num_channels_down) 

    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)) :
        upsample_mode   = [upsample_mode]*n_scales

    if not (isinstance(downsample_mode, list)or isinstance(downsample_mode, tuple)):
        downsample_mode   = [downsample_mode]*n_scales
    
    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)) :
        filter_size_down   = [filter_size_down]*n_scales

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
        filter_size_up   = [filter_size_up]*n_scales

    last_scale = n_scales - 1 

    cur_depth = None

    model = nn.Sequential()
    model_tmp = model

    input_depth = num_input_channels
    for i in range(len(num_channels_down)):

        deeper = nn.Sequential()
        skip = nn.Sequential()

        if num_channels_skip[i] != 0:
            model_tmp.add_module(f"model_concat_{i}" ,Concat(1, skip, deeper))
        else:
            model_tmp.add_module(f"model_deeper_{i}" ,deeper)
        
        model_tmp.add_module(f"bn_{i}",bn(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])))

        if num_channels_skip[i] != 0:
            skip.add_module(f"skip_conv_{i}",conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
            skip.add_module(f"skip_bn_{i}",bn(num_channels_skip[i]))
            skip.add_module(f"skip_act_{i}",act(act_fun))
            
        # skip.add(Concat(2, GenNoise(nums_noise[i]), skip_part))

        deeper.add_module(f"deeper_conv_1_{i}",conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[i]))
        deeper.add_module(f"deeper_bn_1_{i}",bn(num_channels_down[i]))
        deeper.add_module(f"deeper_act_1_{i}",act(act_fun))

        deeper.add_module(f"deeper_conv_2_{i}",conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
        deeper.add_module(f"deeper_bn_2_{i}",bn(num_channels_down[i]))
        deeper.add_module(f"deeper_act_2_{i}",act(act_fun))

        deeper_main = nn.Sequential()

        if i == len(num_channels_down) - 1:
            # The deepest
            k = num_channels_down[i]
        else:
            deeper.add_module(f"deeper_main",deeper_main)
            k = num_channels_up[i + 1]

        deeper.add_module(f"upsample_{i}",nn.Upsample(scale_factor=2, mode=upsample_mode[i], align_corners=False))

        model_tmp.add_module(f"model_conv_{i}", conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
        model_tmp.add_module(f"model_bn_{i}", bn(num_channels_up[i]))
        model_tmp.add_module(f"model_act_{i}", act(act_fun))


        if need1x1_up:
            model_tmp.add_module(f"model_conv_1_{i}",conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
            model_tmp.add_module(f"model_bn_1_{i}",bn(num_channels_up[i]))
            model_tmp.add_module(f"model_act_1_{i}",act(act_fun))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main

    model.add_module(f"final_conv", conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))
    if need_sigmoid:
        model.add_module(f"final_sigmoid",nn.Sigmoid())

    return model

##### end conv2d #####
    
def skip1d(
        num_input_channels=2, num_output_channels=3, 
        num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], num_channels_skip=[4, 4, 4, 4, 4], 
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        need_sigmoid=True, need_bias=True, 
        pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU', 
        need1x1_up=True):
    """Assembles encoder-decoder with skip connections.

    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

    """
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

    n_scales = len(num_channels_down) 

    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)) :
        upsample_mode   = [upsample_mode]*n_scales

    if not (isinstance(downsample_mode, list)or isinstance(downsample_mode, tuple)):
        downsample_mode   = [downsample_mode]*n_scales
    
    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)) :
        filter_size_down   = [filter_size_down]*n_scales

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
        filter_size_up   = [filter_size_up]*n_scales

    last_scale = n_scales - 1 

    cur_depth = None

    model = nn.Sequential()
    model_tmp = model

    input_depth = num_input_channels
    for i in range(len(num_channels_down)):

        deeper = nn.Sequential()
        skip = nn.Sequential()

        if num_channels_skip[i] != 0:
            model_tmp.add_module(f"concat_{i}", Concat1d(1, skip, deeper))
        else:
            model_tmp.add_module(f"deeper_{i}",deeper)
        
        model_tmp.add_module(f"batchnorm_{i}", bn1d(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])))

        if num_channels_skip[i] != 0:
            skip.add_module(f"skip_convolution1d_{i}",conv1d(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
            skip.add_module(f"skip_batchnorm_{i}",bn1d(num_channels_skip[i]))
            skip.add_module(f"skip_activation_{i}",act(act_fun))
            
        # skip.add(Concat(2, GenNoise(nums_noise[i]), skip_part))

        deeper.add_module(f"deeper_convolution1d_1_{i}",conv1d(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[i]))
        deeper.add_module(f"deeper_batchnorm_1_{i}",bn1d(num_channels_down[i]))
        deeper.add_module(f"deeper_activation_1_{i}",act(act_fun))

        deeper.add_module(f"deeper_convolution_2_{i}",conv1d(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
        deeper.add_module(f"deeper_batchnorm_2_{i}",bn1d(num_channels_down[i]))
        deeper.add_module(f"deeper_activation_2_{i}",act(act_fun))

        deeper_main = nn.Sequential()

        if i == len(num_channels_down) - 1:
            # The deepest
            k = num_channels_down[i]
        else:
            deeper.add_module(f"deeper_main_{i}",deeper_main)
            k = num_channels_up[i + 1]

        deeper.add_module(f"deeper_upsample_{i}", nn.Upsample(scale_factor=2, mode=upsample_mode[i]))

        model_tmp.add_module(f"upsampling_convolution_1_{i}", conv1d(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
        model_tmp.add_module(f"upsampling_batchnorm_1_{i}", bn1d(num_channels_up[i]))
        model_tmp.add_module(f"upsampling_activation_1_{i}", act(act_fun))


        if need1x1_up:
            model_tmp.add_module(f"upsampling_convolution_2_{i}", conv1d(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
            model_tmp.add_module(f"upsampling_convolution_2_{i}", bn1d(num_channels_up[i]))
            model_tmp.add_module(f"upsampling_convolution_2_{i}", act(act_fun))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main

    model.add_module(f"final", conv1d(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))
    if need_sigmoid:
        model.add(nn.Sigmoid())

    return model

def skip1d_dropout(
        num_input_channels=2, num_output_channels=3, 
        num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], num_channels_skip=[4, 4, 4, 4, 4], 
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        need_sigmoid=True, need_bias=True, 
        pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU', dropout_rate=0.3, 
        need1x1_up=True):
    """Assembles encoder-decoder with skip connections.

    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

    """
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

    n_scales = len(num_channels_down) 

    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)) :
        upsample_mode   = [upsample_mode]*n_scales

    if not (isinstance(downsample_mode, list)or isinstance(downsample_mode, tuple)):
        downsample_mode   = [downsample_mode]*n_scales
    
    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)) :
        filter_size_down   = [filter_size_down]*n_scales

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
        filter_size_up   = [filter_size_up]*n_scales

    last_scale = n_scales - 1 

    cur_depth = None

    model = nn.Sequential()
    model_tmp = model

    input_depth = num_input_channels
    for i in range(len(num_channels_down)):

        deeper = nn.Sequential()
        skip = nn.Sequential()

        if num_channels_skip[i] != 0:
            model_tmp.add_module(f"concat_{i}", Concat1d(1, skip, deeper))
        else:
            model_tmp.add_module(f"deeper_{i}",deeper)
        
        model_tmp.add_module(f"batchnorm_{i}", bn1d(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])))

        if num_channels_skip[i] != 0:
            skip.add_module(f"skip_convolution1d_{i}",conv1d(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
            skip.add_module(f"skip_batchnorm_{i}",bn1d(num_channels_skip[i]))
            skip.add_module(f"skip_activation_{i}",act(act_fun))
            
        # skip.add(Concat(2, GenNoise(nums_noise[i]), skip_part))

        deeper.add_module(f"deeper_convolution1d_1_{i}",conv1d(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[i]))
        deeper.add_module(f"deeper_batchnorm_1_{i}",bn1d(num_channels_down[i]))
        deeper.add_module(f"deeper_activation_1_{i}",act(act_fun))

        deeper.add_module(f"deeper_convolution_2_{i}",conv1d(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
        deeper.add_module(f"deeper_batchnorm_2_{i}",bn1d(num_channels_down[i]))
        deeper.add_module(f"deeper_activation_2_{i}",act(act_fun))

        deeper_main = nn.Sequential()

        if i == len(num_channels_down) - 1:
            # The deepest
            k = num_channels_down[i]
        else:
            deeper.add_module(f"deeper_main_{i}",deeper_main)
            k = num_channels_up[i + 1]

        deeper.add_module(f"deeper_upsample_{i}", nn.Upsample(scale_factor=2, mode=upsample_mode[i]))

        model_tmp.add_module(f"upsampling_dropout_1_bs_{i}", nn.Dropout(p=dropout_rate))
        model_tmp.add_module(f"upsampling_convolution_1_{i}", conv1d(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
        model_tmp.add_module(f"upsampling_batchnorm_1_{i}", bn1d(num_channels_up[i]))
        model_tmp.add_module(f"upsampling_activation_1_{i}", act(act_fun))


        if need1x1_up:
            model_tmp.add_module(f"upsampling_dropout_2_bs_{i}", nn.Dropout(p=dropout_rate))
            model_tmp.add_module(f"upsampling_convolution_2_{i}", conv1d(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
            model_tmp.add_module(f"upsampling_convolution_2_{i}", bn1d(num_channels_up[i]))
            model_tmp.add_module(f"upsampling_convolution_2_{i}", act(act_fun))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main

    model.add_module(f"final", conv1d(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))
    if need_sigmoid:
        model.add(nn.Sigmoid())

    return model

class BlindSpot1D(nn.Module):
    def __init__(self, in_channels, blind_conv_channels=16, depth=4, kernel_size=3, bs_size=1):
        super().__init__()
        self.depth = depth
        self.bs_size = bs_size
        self.blind_conv_channels = blind_conv_channels

        self.blind_convs = nn.ModuleList()
        self.scalars = nn.ParameterList()

        for d in range(depth):
            dilation = 2 ** d
            if d == depth - 1:
                dilation += bs_size // 2  # Increase dilation for last layer

            pad_amt = dilation * (kernel_size // 2)

            c_in = in_channels if d == 0 else blind_conv_channels

            self.blind_convs.append(
                ConvHole1D(
                    in_channels=c_in,
                    out_channels=blind_conv_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation,
                    bias=True
                )
            )
            self.blind_convs.append(nn.ReLU())

            if d < depth - 1:
                self.scalars.append(nn.Parameter(torch.ones(in_channels), requires_grad=True))

    def forward(self, x):
        x1 = x
        for d in range(self.depth):
            if d > 0:
                x1 = x1 + (self.scalars[d - 1].view(1, -1, 1) * x)
            x1 = self.blind_convs[2 * d](x1)
            x1 = self.blind_convs[2 * d + 1](x1)
        return x1
    

def skip1d_bs(
        num_input_channels=2, num_output_channels=3, 
        num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], num_channels_skip=[4, 4, 4, 4, 4], 
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        need_sigmoid=True, need_bias=True, 
        pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU', blind_conv_channels=8,
        need1x1_up=True):
    """Assembles encoder-decoder with skip connections.

    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

    """
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

    n_scales = len(num_channels_down) 

    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)) :
        upsample_mode   = [upsample_mode]*n_scales

    if not (isinstance(downsample_mode, list)or isinstance(downsample_mode, tuple)):
        downsample_mode   = [downsample_mode]*n_scales
    
    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)) :
        filter_size_down   = [filter_size_down]*n_scales

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
        filter_size_up   = [filter_size_up]*n_scales

    last_scale = n_scales - 1 

    cur_depth = None

    model = nn.Sequential()
    model_tmp = model

    input_depth = num_input_channels
    for i in range(len(num_channels_down)):

        deeper = nn.Sequential()
        skip = nn.Sequential()

        if num_channels_skip[i] != 0:
            model_tmp.add_module(f"concat_bs_{i}", Concat1d(1, skip, deeper))
        else:
            model_tmp.add_module(f"deeper_bs_{i}",deeper)
        
        model_tmp.add_module(f"batchnorm_bs_{i}", bn1d(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])))

        if num_channels_skip[i] != 0:
            skip.add_module(f"skip_convolution1d_bs_{i}",conv1d(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
            skip.add_module(f"skip_batchnorm_bs_{i}",bn1d(num_channels_skip[i]))
            skip.add_module(f"skip_activation_bs_{i}",act(act_fun))
            
        # skip.add(Concat(2, GenNoise(nums_noise[i]), skip_part))

        deeper.add_module(f"deeper_convolution1d_1_bs_{i}",conv1d(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[i]))
        deeper.add_module(f"deeper_batchnorm_1_bs_{i}",bn1d(num_channels_down[i]))
        deeper.add_module(f"deeper_activation_1_bs_{i}",act(act_fun))

        deeper.add_module(f"deeper_convolution_2_bs_{i}",conv1d(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
        deeper.add_module(f"deeper_batchnorm_2_bs_{i}",bn1d(num_channels_down[i]))
        deeper.add_module(f"deeper_activation_2_bs_{i}",act(act_fun))

        deeper_main = nn.Sequential()

        if i == len(num_channels_down) - 1:
            # The deepest
            k = num_channels_down[i]
        else:
            deeper.add_module(f"deeper_main_bs_{i}",deeper_main)
            k = num_channels_up[i + 1]

        deeper.add_module(f"deeper_upsample_bs_{i}", nn.Upsample(scale_factor=2, mode=upsample_mode[i]))

        model_tmp.add_module(f"upsampling_convolution_1_bs_{i}", conv1d(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
        model_tmp.add_module(f"upsampling_batchnorm_1_bs_{i}", bn1d(num_channels_up[i]))
        model_tmp.add_module(f"upsampling_activation_1_bs_{i}", act(act_fun))


        if need1x1_up:
            model_tmp.add_module(f"upsampling_convolution_2_bs_{i}", conv1d(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
            model_tmp.add_module(f"upsampling_batchnorm_2_bs_{i}", bn1d(num_channels_up[i]))
            model_tmp.add_module(f"upsampling_activation_2_bs_{i}", act(act_fun))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main
    
    #bs network
    blind_spot = BlindSpot1D(in_channels=num_channels_up[0], blind_conv_channels=blind_conv_channels, depth=3, kernel_size=3, bs_size=1)
    model.add_module("blind_spot", blind_spot)

    model.add_module(f"final_bs", conv1d(blind_conv_channels, num_output_channels, 1, bias=need_bias, pad=pad))
    if need_sigmoid:
        model.add(nn.Sigmoid())

    return model


# Unet -> (blindspot3 + blindspot5) -> conv1d 
def build_unet_sequential(num_input_channels, num_channels_down, num_channels_up, num_channels_skip, filter_size_down, filter_size_up, filter_skip_size, need_bias, pad, upsample_mode, downsample_mode, act_fun, need1x1_up, last_scale):
    model = nn.Sequential()
    model_tmp = model

    input_depth = num_input_channels
    for i in range(len(num_channels_down)):

        deeper = nn.Sequential()
        skip = nn.Sequential()

        if num_channels_skip[i] != 0:
            model_tmp.add_module(f"concat_bs_{i}", Concat1d(1, skip, deeper))
        else:
            model_tmp.add_module(f"deeper_bs_{i}",deeper)
        
        model_tmp.add_module(f"batchnorm_bs_{i}", bn1d(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])))

        if num_channels_skip[i] != 0:
            skip.add_module(f"skip_convolution1d_bs_{i}",conv1d(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
            skip.add_module(f"skip_batchnorm_bs_{i}",bn1d(num_channels_skip[i]))
            skip.add_module(f"skip_activation_bs_{i}",act(act_fun))
            
        # skip.add(Concat(2, GenNoise(nums_noise[i]), skip_part))

        deeper.add_module(f"deeper_convolution1d_1_bs_{i}",conv1d(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[i]))
        deeper.add_module(f"deeper_batchnorm_1_bs_{i}",bn1d(num_channels_down[i]))
        deeper.add_module(f"deeper_activation_1_bs_{i}",act(act_fun))

        deeper.add_module(f"deeper_convolution_2_bs_{i}",conv1d(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
        deeper.add_module(f"deeper_batchnorm_2_bs_{i}",bn1d(num_channels_down[i]))
        deeper.add_module(f"deeper_activation_2_bs_{i}",act(act_fun))

        deeper_main = nn.Sequential()

        if i == len(num_channels_down) - 1:
            # The deepest
            k = num_channels_down[i]
        else:
            deeper.add_module(f"deeper_main_bs_{i}",deeper_main)
            k = num_channels_up[i + 1]

        deeper.add_module(f"deeper_upsample_bs_{i}", nn.Upsample(scale_factor=2, mode=upsample_mode[i]))

        model_tmp.add_module(f"upsampling_convolution_1_bs_{i}", conv1d(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
        model_tmp.add_module(f"upsampling_batchnorm_1_bs_{i}", bn1d(num_channels_up[i]))
        model_tmp.add_module(f"upsampling_activation_1_bs_{i}", act(act_fun))


        if need1x1_up:
            model_tmp.add_module(f"upsampling_convolution_2_bs_{i}", conv1d(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
            model_tmp.add_module(f"upsampling_batchnorm_2_bs_{i}", bn1d(num_channels_up[i]))
            model_tmp.add_module(f"upsampling_activation_2_bs_{i}", act(act_fun))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main

    return model

def build_unet_sequential_with_dropout(num_input_channels, num_channels_down, num_channels_up, num_channels_skip, filter_size_down, filter_size_up, filter_skip_size, need_bias, pad, upsample_mode, downsample_mode, act_fun, need1x1_up, last_scale):
    model = nn.Sequential()
    model_tmp = model

    input_depth = num_input_channels
    for i in range(len(num_channels_down)):

        deeper = nn.Sequential()
        skip = nn.Sequential()

        if num_channels_skip[i] != 0:
            model_tmp.add_module(f"concat_bs_{i}", Concat1d(1, skip, deeper))
        else:
            model_tmp.add_module(f"deeper_bs_{i}",deeper)
        
        model_tmp.add_module(f"batchnorm_bs_{i}", bn1d(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])))

        if num_channels_skip[i] != 0:
            skip.add_module(f"skip_convolution1d_bs_{i}",conv1d(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
            skip.add_module(f"skip_batchnorm_bs_{i}",bn1d(num_channels_skip[i]))
            skip.add_module(f"skip_activation_bs_{i}",act(act_fun))
            
        # skip.add(Concat(2, GenNoise(nums_noise[i]), skip_part))

        deeper.add_module(f"deeper_convolution1d_1_bs_{i}",conv1d(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[i]))
        deeper.add_module(f"deeper_batchnorm_1_bs_{i}",bn1d(num_channels_down[i]))
        deeper.add_module(f"deeper_activation_1_bs_{i}",act(act_fun))

        deeper.add_module(f"deeper_convolution_2_bs_{i}",conv1d(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
        deeper.add_module(f"deeper_batchnorm_2_bs_{i}",bn1d(num_channels_down[i]))
        deeper.add_module(f"deeper_activation_2_bs_{i}",act(act_fun))

        deeper_main = nn.Sequential()

        if i == len(num_channels_down) - 1:
            # The deepest
            k = num_channels_down[i]
        else:
            deeper.add_module(f"deeper_main_bs_{i}",deeper_main)
            k = num_channels_up[i + 1]

        deeper.add_module(f"deeper_upsample_bs_{i}", nn.Upsample(scale_factor=2, mode=upsample_mode[i]))

        model_tmp.add_module(f"upsampling_convolution_1_bs_{i}", conv1d(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
        model_tmp.add_module(f"upsampling_batchnorm_1_bs_{i}", bn1d(num_channels_up[i]))
        model_tmp.add_module(f"upsampling_activation_1_bs_{i}", act(act_fun))


        if need1x1_up:
            model_tmp.add_module(f"upsampling_convolution_2_bs_{i}", conv1d(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
            model_tmp.add_module(f"upsampling_batchnorm_2_bs_{i}", bn1d(num_channels_up[i]))
            model_tmp.add_module(f"upsampling_activation_2_bs_{i}", act(act_fun))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main

    return model

class UNetPlusBlindFusion(nn.Module):
    def __init__(self, 
                num_input_channels,
                num_output_channels,
                blind_conv_channels,
                num_channels_down,
                num_channels_up,
                num_channels_skip,
                filter_size_down,
                filter_size_up,
                filter_skip_size,
                need_bias,
                pad,
                upsample_mode,
                downsample_mode,
                act_fun,
                need1x1_up,
                last_scale):
        super().__init__()

        # Step 1: UNet as Sequential
        self.unet_body = build_unet_sequential_with_dropout(
            num_input_channels=num_input_channels,
            num_channels_down = num_channels_down,
            num_channels_up = num_channels_up,
            num_channels_skip = num_channels_skip,
            filter_size_down    = filter_size_down,
            filter_size_up = filter_size_up,
            filter_skip_size    = filter_skip_size,
            need_bias   = need_bias,
            pad = pad,
            upsample_mode  = upsample_mode,
            downsample_mode = downsample_mode,
            act_fun = act_fun,
            need1x1_up  = need1x1_up,
            last_scale = last_scale
        )

        # Step 2: BlindSpot convolutions
        self.blind_spot_3 = BlindSpot1D(
            in_channels=blind_conv_channels,
            blind_conv_channels=blind_conv_channels,
            depth=5, kernel_size=3
        )
        self.blind_spot_5 = BlindSpot1D(
            in_channels=blind_conv_channels,
            blind_conv_channels=blind_conv_channels,
            depth=3, kernel_size=5
        )

        # Step 3: Concatenation projection
        self.fuse = nn.Sequential(
            nn.Conv1d(2 * blind_conv_channels, blind_conv_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(blind_conv_channels, num_output_channels, kernel_size=1)
        )

    def forward(self, x):
        x_unet = self.unet_body(x)

        x_bs3 = self.blind_spot_3(x_unet)
        x_bs5 = self.blind_spot_5(x_unet)

        x_cat = torch.cat([x_bs3, x_bs5], dim=1)  # (B, 2*C, T)
        return self.fuse(x_cat)

def skip1d_fuse_bs(
        num_input_channels=2, num_output_channels=3, 
        num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], num_channels_skip=[4, 4, 4, 4, 4], 
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        need_sigmoid=True, need_bias=True, 
        pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU', blind_conv_channels=8,
        need1x1_up=True):
    """Assembles encoder-decoder with skip connections.

    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

    """
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

    n_scales = len(num_channels_down) 

    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)) :
        upsample_mode   = [upsample_mode]*n_scales

    if not (isinstance(downsample_mode, list)or isinstance(downsample_mode, tuple)):
        downsample_mode   = [downsample_mode]*n_scales
    
    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)) :
        filter_size_down   = [filter_size_down]*n_scales

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
        filter_size_up   = [filter_size_up]*n_scales

    last_scale = n_scales - 1 

    cur_depth = None
    
    model = UNetPlusBlindFusion(
        num_input_channels=num_input_channels,
        num_output_channels=num_output_channels,
        blind_conv_channels=blind_conv_channels,
        num_channels_down=num_channels_down,
        num_channels_up=num_channels_up,
        num_channels_skip=num_channels_skip,
        filter_size_down=filter_size_down,
        filter_size_up=filter_size_up,
        filter_skip_size=filter_skip_size,
        need_bias=need_bias,
        pad=pad,
        upsample_mode=upsample_mode,
        downsample_mode=downsample_mode,
        act_fun=act_fun,
        need1x1_up=need1x1_up,
        last_scale=last_scale
    )

    return model


def get_1d_posencode_inp(H, n_inputs):
    '''
        Get positionally encoded inputs for inpainting tasks (CPU version)
        Based on https://bmild.github.io/fourfeat/
    '''
    X = 10 * np.arange(H).reshape(-1, 1) / H
    freqs = np.random.randn(1, n_inputs)
    angles = X.dot(freqs)
    
    sin_vals = np.sin(2 * np.pi * angles)
    cos_vals = np.cos(2 * np.pi * angles)
    
    posencode_vals = np.hstack((sin_vals, cos_vals)).astype(np.float32)
    inp = posencode_vals.reshape(H, 2 * n_inputs)
    
    # Create a trainable parameter on CPU
    inp = torch.nn.Parameter(torch.tensor(inp).permute(1, 0)[None, ...])
    return inp

class L2Norm():
    def __init__(self):
        pass
    def __call__(self, x):
        return (x.pow(2)).mean()
    

class Constant1D(nn.Module):
    def __init__(self, length):
        super().__init__()
        # single learnable scalar shared across all positions
        self.bias = nn.Parameter(torch.tensor(1.0))  
        self.length = length

    def forward(self, _inp):
        # output shape: [length, 1], all values = self.bias
        return self.bias.expand(self.length, 1)