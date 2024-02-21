import numpy as np


class PointFeatureEncoder(object):
    """
    该类决定使用点的哪些属性 比如x,y,z等
    """
    def __init__(self, config, point_cloud_range=None):
        # 初始化方法，用于创建类的实例
        super().__init__()  # 调用父类的初始化方法

        # 将传递给构造函数的 config 参数存储在实例变量 point_encoding_config 中
        self.point_encoding_config = config

        # 使用 assert 语句检查 src_feature_list 的前三个元素是否分别为 'x', 'y', 'z'，否则引发 AssertionError
        assert list(self.point_encoding_config.src_feature_list[0:3]) == ['x', 'y', 'z']

        # 将 point_encoding_config 中的 used_feature_list 赋值给实例变量 used_feature_list
        self.used_feature_list = self.point_encoding_config.used_feature_list  # ['x', 'y', 'z', 'intensity']

        # 将 point_encoding_config 中的 src_feature_list 赋值给实例变量 src_feature_list
        self.src_feature_list = self.point_encoding_config.src_feature_list  # ['x', 'y', 'z', 'intensity']

        # 将传递给构造函数的 point_cloud_range 参数存储在实例变量 point_cloud_range 中
        self.point_cloud_range = point_cloud_range

    # 让你可以直接用 类名.num_point_features 的方式来得到这个属性，等于把函数变成属性了
    @property
    def num_point_features(self):
        # 使用 getattr 获取对象属性，调用指定的点编码方法（absolute_coordinates_encoding）来计算特征数量
        # 这里传递的 points 参数为 None，目的是获取特征数量而不是实际的点特征
        return getattr(self, self.point_encoding_config.encoding_type)(points=None)  # 返回特征数量，这里为 4


    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)  # 输入数据，可能包含点云坐标以及其他特征
                ...
        Returns:
            data_dict:
                points: (N, 3 + C_out),  # 处理后的数据，可能更新了点云坐标或特征
                use_lead_xyz: whether to use xyz as point-wise features  # 指示是否使用坐标作为点特征的标志
                ...
        """
        # 调用 self.point_encoding_config.encoding_type 对应的函数，处理输入的点云数据
        # 例如，可能会对点云坐标进行编码，产生新的特征表示
        # 返回的是处理后的点云数据以及一个标志 use_lead_xyz，表示是否使用坐标作为点特征
        data_dict['points'], use_lead_xyz = getattr(self, self.point_encoding_config.encoding_type)(
            data_dict['points']
        )
        
        # 将标志 use_lead_xyz 存储到 data_dict 中
        data_dict['use_lead_xyz'] = use_lead_xyz
        
        # 返回处理后的 data_dict
        return data_dict


    def absolute_coordinates_encoding(self, points=None):
        # 如果 points 为 None，返回用于表示输出特征数量的值
        if points is None:
            num_output_features = len(self.used_feature_list)
            return num_output_features

        # 创建一个列表，其中包含点的 x、y、z 坐标
        point_feature_list = [points[:, 0:3]]  # (1, N, 3 + C_in)

        # 遍历 used_feature_list 中的元素
        for x in self.used_feature_list:
            # 如果特征为 'x'、'y' 或 'z'，跳过当前循环
            if x in ['x', 'y', 'z']:
                continue

            # 获取当前特征在 src_feature_list 中的索引
            idx = self.src_feature_list.index(x)  # 3

            # 将当前特征的列添加到 point_feature_list
            point_feature_list.append(points[:, idx:idx+1])  # [(1, N, 3), (N, 1)]

        # 沿着轴 1 连接 point_feature_list 中的所有特征列
        point_features = np.concatenate(point_feature_list, axis=1)  # (N, 4)

        # 返回点的特征和 True（表示成功执行）
        return point_features, True
