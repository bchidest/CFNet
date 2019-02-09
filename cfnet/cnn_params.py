import math
import csv


class CNNParams:

    def __init__(self, hyperparam_dict, conv_layers,
                 fully_connected_layers, transition_layer, side_info_layer,
                 param_file_line_number):
        """
        Order:
            learning_rate - float
            layer_list - list
                layer_type - {"fully", "conv"}
                n_nodes - int


        """
        self.learning_rate = hyperparam_dict['learning_rate']
        assert self.learning_rate > 0
        self.batch_size = hyperparam_dict['batch_size']
        assert self.batch_size > 0
        self.n_classes = hyperparam_dict['n_classes']
        assert self.n_classes > 0
        self.max_steps = hyperparam_dict['max_steps']
        assert self.max_steps > 0
        self.dropout_prob = hyperparam_dict['dropout_prob']
        assert self.dropout_prob > 0 and self.dropout_prob <= 1
        self.spatial_transform = hyperparam_dict['spatial_transform']
        self.param_file_line_number = param_file_line_number
        self.conv_layers = conv_layers
        self.fully_connected_layers = fully_connected_layers
        self.transition_layer = transition_layer
        self.side_info_layer = side_info_layer

    def calculate_n_output_conv_nodes(self, input_size, ind=-1):
        if ind == -1:
            ind = len(self.conv_layers)
        print(ind)
        for layer in self.conv_layers[0:ind]:
            input_size = layer.calculate_n_output_size(input_size)
        return input_size**2 * layer.n_filters

    def calculate_conv_feature_map_dims(self, input_size):
        dims = []
        for layer in self.conv_layers:
            input_size = layer.calculate_n_output_size(input_size)
            dims.append(input_size)
        return dims

    def calculate_conv_remainders(self, input_size):
        # TODO: This is hardcoded for a stride of 2 only!
        remainders = []
        for layer in self.conv_layers:
            # TODO: Should handle conv stride and pooling seperately
            if (input_size % 2 == 0) and (layer.convolution_stride == 2):
                remainders.append(1)
            elif (input_size % 2 != 0) and (layer.pooling_stride == 2):
                remainders.append(1)
            else:
                remainders.append(0)
            input_size = layer.calculate_n_output_size(input_size)
        return remainders

    def __repr__(self):
        return "CNNParams()"

    def __str__(self):
        s = 'CNNParams'\
                + '\n    learning_rate=' + str(self.learning_rate)\
                + '\n    batch_size=' + str(self.batch_size)\
                + '\n    n_classes=' + str(self.n_classes)\
                + '\n    max_steps=' + str(self.max_steps)\
                + '\n    dropout_prob=' + str(self.n_classes)\
                + '\n    spatial_transform=' + str(self.spatial_transform) + '\n'
        if self.conv_layers:
            for layer in self.conv_layers:
                s += str(layer)
        if self.transition_layer:
            s += str(self.transition_layer)
        if self.fully_connected_layers:
            for layer in self.fully_connected_layers:
                s += str(layer)
        return s


class FullyConnectedLayer:
    def __init__(self, layer_dict):
        self.n_nodes = layer_dict['n_nodes']
        assert self.n_nodes > 0
        self.dropout = layer_dict['dropout']
        self.batch_norm = layer_dict['bn']

    def __str__(self):
        s = '  FullyConnectedLayer'\
                + '\n    n_nodes=' + str(self.n_nodes)\
                + '\n    batch_norm=' + str(self.batch_norm)\
                + '\n    dropout=' + str(self.dropout) + '\n'
        return s


class TransitionLayer:
    def __init__(self, layer_dict, prev_conv_layer):
        self.n_nodes = layer_dict['n_nodes']
        self.dropout = layer_dict['dropout']
        self.n_rotations = layer_dict['n_rotations']
        if isinstance(prev_conv_layer, REConvolutionalLayer):
            assert (self.n_rotations > 0 and
                    self.n_rotations % 4 == 0)
            assert self.n_nodes > 0
        if isinstance(prev_conv_layer, GEConvolutionalLayer):
            assert not prev_conv_layer.ge_pool
        self.batch_norm = layer_dict['bn']

    def __str__(self):
        s = '  TransitionLayer'\
                + '\n    n_nodes=' + str(self.n_nodes)\
                + '\n    batch_norm=' + str(self.batch_norm)\
                + '\n    dropout=' + str(self.dropout)\
                + '\n    n_rotations=' + str(self.n_rotations) + '\n'
        return s


class ConvolutionalLayer(object):
    def __init__(self, layer_dict):
        NotImplementedError("Class %s does not implement __init__()" % (self.__class__.__name__))

    def load_layer_dict(self, layer_dict):
        self.n_filters = layer_dict['n_filts']
        assert self.n_filters > 0
        self.filter_size = layer_dict['filt_size']
        assert self.filter_size > 0
        self.pooling = layer_dict['pool']
        assert self.pooling in [None, 'max', 'avg']
        self.pooling_support = layer_dict['pool_support']
        self.pooling_stride = layer_dict['pool_stride']
        if self.pooling:
            assert (self.pooling_stride > 0 and self.pooling_support > 1)
        self.convolution_stride = layer_dict['stride']
        assert self.convolution_stride > 0
        self.batch_norm = layer_dict['bn']
        self.padding = layer_dict['padding']
        assert self.padding in ['SAME', 'VALID']

    def calculate_n_output_size(self, input_size):
        if self.padding == 'SAME':
            after_conv_size = int(math.ceil(input_size /
                                  float(self.convolution_stride)))
        else:
            after_conv_size = int(math.ceil((input_size - self.filter_size + 1) /
                                  float(self.convolution_stride)))
        if self.pooling_support > 1:
            if self.pooling_stride > 1:
                after_conv_size = after_conv_size / self.pooling_stride
        return after_conv_size


class StandardConvolutionalLayer(ConvolutionalLayer):
    def __init__(self, layer_dict):
        self.load_layer_dict(layer_dict)

    def __str__(self):
        s = '  StandardConvolutionalLayer'\
                + '\n    n_filters=' + str(self.n_filters)\
                + '\n    filter_size=' + str(self.filter_size)\
                + '\n    pooling=' + str(self.pooling)\
                + '\n    pooling_support=' + str(self.pooling_support)\
                + '\n    pooling_stride=' + str(self.pooling_stride)\
                + '\n    convolution_stride=' + str(self.convolution_stride)\
                + '\n    batch_norm=' + str(self.batch_norm)\
                + '\n    padding=' + self.padding + '\n'
        return s


def calculate_reconv_midpoints(x_s, W_radius, convolution_stride, padding):
    if padding == 'SAME':
        stride_remainder = (x_s / 2) % convolution_stride
    else:
        stride_remainder = (x_s / 2 - W_radius - 1) % convolution_stride
    mid = x_s / 2
    if x_s % 2 == 0:
        mid_a = mid
        mid_b = mid + stride_remainder
    else:
        mid_b = mid + 1 + stride_remainder
        mid_a = mid - stride_remainder
    return mid_a, mid_b


class REConvolutionalLayer(ConvolutionalLayer):
    def __init__(self, layer_dict):
        self.load_layer_dict(layer_dict)
        self.load_layer_dict_reconv(layer_dict)

    def load_layer_dict_reconv(self, layer_dict):
        self.n_rotations = layer_dict['n_rotations']
        assert self.n_rotations in [4, 8]

    def calculate_midpoints(self, input_size):
        return calculate_reconv_midpoints(
                input_size, int(self.filter_size / 2),
                self.convolution_stride, self.padding)

    def calculate_n_output_size(self, input_size):
        x_s = int(input_size)
        W_radius = int(self.filter_size / 2)
        mid_a, _ = self.calculate_midpoints(x_s)
        after_conv_size = 2*int(math.ceil((mid_a - W_radius) /
                                float(self.convolution_stride)))
        if x_s % 2 != 0:
            after_conv_size += 1
        if self.pooling:
            after_conv_size = after_conv_size / self.pooling_stride
        return after_conv_size

    def __str__(self):
        s = '  REConvolutionalLayer'\
                + '\n    n_filters=' + str(self.n_filters)\
                + '\n    filter_size=' + str(self.filter_size)\
                + '\n    pooling=' + str(self.pooling)\
                + '\n    pooling_support=' + str(self.pooling_support)\
                + '\n    pooling_stride=' + str(self.pooling_stride)\
                + '\n    convolution_stride=' + str(self.convolution_stride)\
                + '\n    batch_norm=' + str(self.batch_norm)\
                + '\n    padding=' + self.padding\
                + '\n    n_rotations=' + str(self.n_rotations) + '\n'
        return s


class GEConvolutionalLayer(ConvolutionalLayer):
    def __init__(self, layer_dict):
        self.load_layer_dict(layer_dict)
        self.ge_type = layer_dict['ge_type']
        assert self.ge_type in ['Z2', 'C4', 'D4']
        self.ge_pool = layer_dict['ge_pool']
        assert self.ge_pool in [None, 'max', 'avg']

    def __str__(self):
        s = '  GEConvolutionalLayer'\
                + '\n    n_filters=' + str(self.n_filters)\
                + '\n    filter_size=' + str(self.filter_size)\
                + '\n    pooling=' + str(self.pooling)\
                + '\n    pooling_support=' + str(self.pooling_support)\
                + '\n    pooling_stride=' + str(self.pooling_stride)\
                + '\n    convolution_stride=' + str(self.convolution_stride)\
                + '\n    batch_norm=' + str(self.batch_norm)\
                + '\n    padding=' + self.padding\
                + '\n    ge_type=' + self.ge_type\
                + '\n    ge_pool=' + str(self.ge_pool) + '\n'
        return s


def reset_hyperparams():
    hyperparam_dict = {
            'learning_rate': 0.01,
            'n_classes': 0,
            'batch_size': 50,
            'max_steps': 1000,
            'dropout_prob': 1.0,
            'spatial_transform': False}
    return hyperparam_dict


def load_params(param_filename):
    param_file = open(param_filename, "r")
    param_csv = csv.reader(param_file, delimiter=",")
    param = []
    conv_layers = []
    full_layers = []
    transition_layer = None
    side_info_layer = None
    line_number = 0
    param_file_line_number = line_number
    hyperparam_dict = reset_hyperparams()
    for line in param_csv:
        if not line:
            # Empty line signals end of model instance
            param.append(CNNParams(hyperparam_dict, conv_layers,
                                   full_layers, transition_layer,
                                   side_info_layer,
                                   param_file_line_number))
            line_number = line_number + 1
            param_file_line_number = line_number
            hyperparam_dict = reset_hyperparams()
            conv_layers = []
            full_layers = []
            transition_layer = None
            side_info_layer = None
            continue
        line_name = line[0].split('=')
        if line_name[0] == 'lr':
            hyperparam_dict['learning_rate'] = float(line_name[1])
            continue
        elif line_name[0] == "n_classes":
            hyperparam_dict['n_classes'] = int(line_name[1])
            continue
        elif line_name[0] == "batch_size":
            hyperparam_dict['batch_size'] = int(line_name[1])
            continue
        elif line_name[0] == "max_steps":
            hyperparam_dict['max_steps'] = int(line_name[1])
            continue
        elif line_name[0] == "dropout_prob":
            hyperparam_dict['dropout_prob'] = float(line_name[1])
            continue
        elif line_name[0] == "spatial_transform":
            hyperparam_dict['spatial_transform'] = (line_name[1] == 'True')
            continue
        elif line_name[0] == '':
            continue
        elif (line_name[0] == 'conv' or
                line_name[0] == 'conv-re' or
                line_name[0] == 'conv-ge' or
                line_name[0] == 'full' or
                line_name[0] == 'dft'):
            layer_dict = {
                    'n_filts': None,
                    'n_nodes': 0,
                    'filt_size': 3,
                    'pool': None,
                    'pool_support': -1,
                    'pool_stride': -1,
                    'stride': 1,
                    'n_rotations': 0,
                    'ge_type': None,
                    'ge_pool': None,
                    'bn': False,
                    'padding': 'VALID',
                    'dropout': False}
            for s in line[1:]:
                s_name = s.split('=')
                if s_name[0] == 'n_filts':
                    layer_dict[s_name[0]] = int(s_name[1])
                elif s_name[0] == 'filt_size':
                    layer_dict[s_name[0]] = int(s_name[1])
                elif s_name[0] == 'pool':
                    layer_dict[s_name[0]] = s_name[1]
                elif s_name[0] == 'pool_support':
                    layer_dict[s_name[0]] = int(s_name[1])
                elif s_name[0] == 'pool_stride':
                    layer_dict[s_name[0]] = int(s_name[1])
                elif s_name[0] == 'stride':
                    layer_dict[s_name[0]] = int(s_name[1])
                elif s_name[0] == 'n_rotations':
                    layer_dict[s_name[0]] = int(s_name[1])
                elif s_name[0] == 'ge_type':
                    layer_dict[s_name[0]] = s_name[1]
                elif s_name[0] == 'ge_pool':
                    layer_dict[s_name[0]] = s_name[1]
                elif s_name[0] == 'n_nodes':
                    layer_dict[s_name[0]] = int(s_name[1])
                elif s_name[0] == 'bn':
                    layer_dict[s_name[0]] = (s_name[1] == 'True')
                elif s_name[0] == 'padding':
                    layer_dict[s_name[0]] = s_name[1]
                elif s_name[0] == 'dropout':
                    layer_dict[s_name[0]] = (s_name[1] == 'True')
            if line_name[0] == 'conv':
                conv_layers.append(StandardConvolutionalLayer(layer_dict))
            elif line_name[0] == 'conv-re':
                conv_layers.append(REConvolutionalLayer(layer_dict))
            elif line_name[0] == 'conv-ge':
                conv_layers.append(GEConvolutionalLayer(layer_dict))
            elif line_name[0] == 'full':
                full_layers.append(FullyConnectedLayer(layer_dict))
            elif line_name[0] == 'dft':
                transition_layer = TransitionLayer(layer_dict, conv_layers[-1])
            else:
                raise ValueError('Unknown parameter name \'' + line_name[0] + '\'!')
        else:
            raise ValueError('Unknown parameter name \'' + line_name[0] + '\'!')
    # EOF
    param.append(CNNParams(hyperparam_dict, conv_layers,
                           full_layers, transition_layer,
                           side_info_layer,
                           param_file_line_number))
    return param
