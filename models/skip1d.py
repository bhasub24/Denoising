import pdb

import torch
import torch.nn as nn
import numpy as np

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

def bn1d(num_features):
    return nn.BatchNorm1d(num_features)

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