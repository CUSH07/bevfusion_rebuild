import torch.nn as nn
import spconv

"""
体素编码后的点云进行特征提取，为避免对大量空体素的无效计算，节省内存。这里进行稀疏卷积操作。
grid 与 Voxel的区别
Grid（网格）：

网格通常指的是空间被均匀划分的结构，可以是二维的（如图像像素）或三维的（如三维网格）。
在二维空间中，网格由行和列组成，每个单元格代表空间中的一个点。
在三维空间中，网格由多层二维网格组成，类似于多层图像。
网格用于表示空间中的均匀划分，每个网格单元具有相同的尺寸和形状。
Voxel（体素）：

体素是三维空间中的基本单位，类似于二维空间中的像素。
体素代表三维空间中的一个小立方体，用于表示三维物体或场景。
体素化是将连续的三维空间离散化为一系列体素的过程，这样每个体素可以独立地存储信息（如颜色、密度、标签等）。
在点云处理中，体素化通常用于将稀疏的点云数据转换为更规则的三维网格形式，以便于应用标准的三维卷积等操作。

voxel_features 是一个张量，用于存储每个体素中点云的特征。
(num_voxels, C)， num_voxels 是体素的总数，C 是每个体素的特征维度。

voxel_coords 是一个包含体素坐标信息的张量，用于描述点云数据在体素化（voxelization）过程中每个体素的位置。
(num_voxels, 4)，num_voxels 是体素的总数。每一行代表一个体素的坐标，[batch_idx, z_idx, y_idx, x_idx]
"""

# VoxelResBackBone8x 类处理体素化后的点云数据


class VoxelResBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size):
        super(VoxelResBackBone8x, self).__init__()
        self.model_cfg = model_cfg

        # 是否使用偏置项和批量归一化函数
        use_bias = self.model_cfg.get('USE_BIAS', False)
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        # 稀疏张量的空间形状
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        # 构建网络的输入层
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU()
        )

        # 构建网络的多层卷积块
        self.conv1 = self.create_conv_block(16, 16, use_bias, norm_fn, 'res1')
        self.conv2 = self.create_conv_block(16, 32, use_bias, norm_fn, 'spconv2', stride=2)
        self.conv3 = self.create_conv_block(32, 64, use_bias, norm_fn, 'spconv3', stride=2)
        self.conv4 = self.create_conv_block(64, 128, use_bias, norm_fn, 'spconv4', stride=2)

        # 构建网络的输出层
        self.conv_out = spconv.SparseSequential(
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=0, bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU()
        )

        # 输出特征的数量
        self.num_point_features = 128

    def create_conv_block(self, in_channels, out_channels, use_bias, norm_fn, indice_key, stride=1):
        block = spconv.SparseSequential(
            SparseBasicBlock(in_channels, out_channels, bias=use_bias, norm_fn=norm_fn, indice_key=indice_key),
            SparseBasicBlock(out_channels, out_channels, bias=use_bias, norm_fn=norm_fn, indice_key=indice_key)
        )
        if stride > 1:
            block.add_module('downsample', spconv.SparseConv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False, indice_key=indice_key))
        return block

    def forward(self, batch_dict):
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']

        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        out = self.conv_out(x_conv4)

        # 更新 batch_dict
        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8,
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            },
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        return batch_dict


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, bias=None, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        if bias is None:
            bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out

def replace_feature(out, new_features):
    if "replace_feature" in out.__dir__():
        # spconv 2.x behaviour
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out
class partial:
    """New function with partial application of the given arguments
    and keywords.
    """

    __slots__ = "func", "args", "keywords", "__dict__", "__weakref__"

    def __new__(*args, **keywords):
        if not args:
            raise TypeError("descriptor '__new__' of partial needs an argument")
        if len(args) < 2:
            raise TypeError("type 'partial' takes at least one argument")
        cls, func, *args = args
        if not callable(func):
            raise TypeError("the first argument must be callable")
        args = tuple(args)

        if hasattr(func, "func"):
            args = func.args + args
            tmpkw = func.keywords.copy()
            tmpkw.update(keywords)
            keywords = tmpkw
            del tmpkw
            func = func.func

        self = super(partial, cls).__new__(cls)

        self.func = func
        self.args = args
        self.keywords = keywords
        return self

    def __call__(*args, **keywords):
        if not args:
            raise TypeError("descriptor '__call__' of partial needs an argument")
        self, *args = args
        newkeywords = self.keywords.copy()
        newkeywords.update(keywords)
        return self.func(*self.args, *args, **newkeywords)

    @recursive_repr()
    def __repr__(self):
        qualname = type(self).__qualname__
        args = [repr(self.func)]
        args.extend(repr(x) for x in self.args)
        args.extend(f"{k}={v!r}" for (k, v) in self.keywords.items())
        if type(self).__module__ == "functools":
            return f"functools.{qualname}({', '.join(args)})"
        return f"{qualname}({', '.join(args)})"

    def __reduce__(self):
        return type(self), (self.func,), (self.func, self.args,
               self.keywords or None, self.__dict__ or None)

    def __setstate__(self, state):
        if not isinstance(state, tuple):
            raise TypeError("argument to __setstate__ must be a tuple")
        if len(state) != 4:
            raise TypeError(f"expected 4 items in state, got {len(state)}")
        func, args, kwds, namespace = state
        if (not callable(func) or not isinstance(args, tuple) or
           (kwds is not None and not isinstance(kwds, dict)) or
           (namespace is not None and not isinstance(namespace, dict))):
            raise TypeError("invalid partial state")

        args = tuple(args) # just in case it's a subclass
        if kwds is None:
            kwds = {}
        elif type(kwds) is not dict: # XXX does it need to be *exactly* dict?
            kwds = dict(kwds)
        if namespace is None:
            namespace = {}

        self.__dict__ = namespace
        self.func = func
        self.args = args
        self.keywords = kwds
def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )  

    return m
