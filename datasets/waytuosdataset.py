import sys
sys.path.append('/root/autodl-fs/pointpillar')
sys.path.append('/root/autodl-fs/pointpillar/datasets')

print(sys.path)


import copy
import pickle

import numpy as np
from skimage import io

# from . import waytuos_utils
from ops.roiaware_pool3d import roiaware_pool3d_utils
from utils import box_utils, calibration_waytuos, common_utils, object3d_waytuos
from dataset import DatasetTemplate


def calib_to_matricies(calib):
    """
    将Calibration对象转换为变换矩阵
    参数:
        calib: calibration.Calibration, Calibration对象
    返回:
        V2R: (4, 4), 激光雷达到矫正后的相机变换矩阵
        P_left: (3, 4), 相机投影矩阵
    """
    # 将相机到车辆坐标系的变换矩阵V2C增加一行 [0, 0, 0, 1]，得到(4, 4)的矩阵
    V2C = np.vstack((calib.V2C, np.array([0, 0, 0, 1], dtype=np.float32)))  # (4, 4)
    
    # 将相机到激光雷达坐标系的旋转矩阵R0增加一列 [0, 0, 0, 1]，得到(3, 4)的矩阵
    R0_rect = np.hstack((calib.R0_rect, np.zeros((3, 1), dtype=np.float32)))  # (3, 4)
    
    # 将R0增加一行 [0, 0, 0, 1]，得到(4, 4)的矩阵
    R0_rect = np.vstack((R0_rect, np.array([0, 0, 0, 1], dtype=np.float32)))  # (4, 4)
    
    # 计算激光雷达到矫正后的相机坐标系的变换矩阵V2R，通过R0乘以V2C得到(4, 4)的矩阵
    V2R = R0_rect @ V2C
    
    # 获取相机投影矩阵P2
    P_left = calib.P_left
    
    # 返回激光雷达到矫正后的相机坐标系的变换矩阵V2R和相机投影矩阵P2
    return V2R, P_left


# 定义waytuos数据集的类
class Waytuos(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        # 初始化类，将参数赋值给类的属性
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        # 传递参数是 训练集train 还是验证集val
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        # root_path的路径是../data/waytuos/
        # waytuos数据集一共三个文件夹“training”和“testing”、“ImageSets”
        # 如果是训练集train，将文件的路径指为训练集training ，否则为测试集testing
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')
        # /data/waytuos/ImageSets/下面一共三个文件：test.txt , train.txt ,val.txt
        # 选择其中的一个文件
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        # 得到.txt文件下的序列号，组成列表sample_id_list
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None
        # 创建用于存放waytuos信息的空列表
        self.waytuos_infos = []
        # 调用函数，加载waytuos数据，mode的值为：train 或者  test
        self.include_waytuos_data(self.mode)

    def include_waytuos_data(self, mode):
        if self.logger is not None:
            # 如果日志信息存在，则加入'Loading waytuos dataset'的信息
            self.logger.info('Loading waytuos dataset')
        # 创建新列表，用于存放信息
        waytuos_infos = []

        '''   
        INFO_PATH: {
        'train': [waytuos_infos_train.pkl],
        'test': [waytuos_infos_val.pkl],}
        '''
        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            # root_path的路径是/data/waytuos/
            info_path = self.root_path / info_path
            # 则 info_path：/data/waytuos/waytuos_infos_train.pkl之类的文件
            if not info_path.exists():
                # 如果该文件不存在，跳出，继续下一个文件
                continue
            # 打开该文件
            with open(info_path, 'rb') as f:
                # pickle.load(f) 将该文件中的数据解析为一个Python对象infos，
                # 并将该内容添加到waytuos_infos列表中
                infos = pickle.load(f)
                waytuos_infos.extend(infos)

        self.waytuos_infos.extend(waytuos_infos)

        if self.logger is not None:
            self.logger.info('Total samples for waytuos dataset: %d' % (len(waytuos_infos)))


    def set_split(self, split):
        """根据指定的数据集划分，获取相应的标签列表

        Args:
            split (string): 数据集划分，可以是'train'或'test'

        Returns:
            list: 标签列表

        """
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        # root_path的路径是/data/waytuos/ 
        # 则root_split_path=/data/waytuos/training
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'val' else 'testing')
        # /data/waytuos/ImageSets/下面一共三个文件：test.txt , train.txt ,val.txt
        # 选择其中的一个文件
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        # 将文件中的tag构造为列表，方便处理
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None


    def get_lidar(self, idx):
        """加载样本的激光雷达点云数据
        Args:
            index (int): 获取点云文件的索引。
        Returns:
            np.array(N, 4): 点云数据。
        """
        # /data/waytuos/training/velodyne/xxxxxx.bin
        lidar_file = self.root_split_path / 'velodyne' / ('%s.bin' % idx)
        assert lidar_file.exists()
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)


    def get_image(self, idx):
        """
        加载样本的图像
        Args:
            idx: int, 样本索引
        Returns:
            image: (H, W, 3), RGB图像
        """
        # 图像文件路径：/data/waytuos/training/image_2/xxxxxx.png
        img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
        assert img_file.exists()
        
        # 使用图像读取库加载图像文件
        image = io.imread(img_file)
        
        # 将图像数据类型转换为32位浮点数
        image = image.astype(np.float32)
        
        # 对图像进行归一化，将像素值范围从[0, 255]缩放到[0, 1]
        image /= 255.0
        
        # 返回表示图像的NumPy数组，形状为(H, W, 3)，其中H和W分别是图像的高度和宽度，3表示RGB通道
        return image


    def get_image_shape(self, idx):
        # 图像文件路径：/data/waytuos/training/image_2/xxxxxx.png
        img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
        
        # 断言确保图像文件存在，否则抛出异常
        assert img_file.exists()
        
        # 使用图像读取库加载图像文件，并获取图像的形状
        # 对于waytuos数据集的图像，返回的形状是包含高度和宽度的数组，例如：array([375, 1242], dtype=int32)
        image_shape = np.array(io.imread(img_file).shape[:2], dtype=np.int32)
        
        # 返回图像的形状
        # 含有高度和宽度的numpy数组
        return image_shape


    def get_label(self, idx):
        # /data/waytuos/training/label_2/xxxxxx.txt
        label_file = self.root_split_path / 'label_2' / ('%s.txt' % idx)
        assert label_file.exists() #如果不存在，直接报错
        # 调用get_objects_from_label函数，首先读取该文件的所有行赋值为 lines
        # 在对lines中的每一个line（一个object的参数）作为object3d类的参数 进行遍历，
        # 最后返回：objects[]列表 ,里面是当前文件里所有物体的属性值，如：type、x,y,等
        return object3d_waytuos.get_objects_from_label(label_file)

    # 用不着
    def get_depth_map(self, idx):
        """
        加载样本的深度图
        Args:
            idx: str, 样本索引
        Returns:
            depth: (H, W), 深度图
        """
        # 深度图文件路径：/data/waytuos/training/depth_2/xxxxxx.txt
        depth_file = self.root_split_path / 'depth_2' / ('%s.png' % idx)
        assert depth_file.exists()  # 如果深度图文件不存在，报错
        
        # 读取深度图文件，并进行数据类型转换和归一化处理
        depth = io.imread(depth_file)
        depth = depth.astype(np.float32)
        depth /= 256.0
        
        # 返回深度图
        return depth


    def get_calib(self, idx):
        # 获取标定文件路径，其中idx是样本索引
        calib_file = self.root_split_path / 'calib' / ('%s.txt' % idx)
        
        # 确保标定文件存在
        assert calib_file.exists()
        
        # 创建Calibration对象并返回
        return calibration_waytuos.Calibration(calib_file)


    # 用不着
    def get_road_plane(self, idx):
        # 获取平面文件路径，其中idx是样本索引
        plane_file = self.root_split_path / 'planes' / ('%s.txt' % idx)
        
        # 如果平面文件不存在，则返回None
        if not plane_file.exists():
            return None

        # 读取平面文件内容
        with open(plane_file, 'r') as f:
            lines = f.readlines()
        
        # 提取平面参数并转换为numpy数组
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # 确保法向量始终朝上，这是在矫正后的相机坐标系中
        if plane[1] > 0:
            plane = -plane

        # 对法向量进行归一化
        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        
        return plane


    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        """ 
        判断点云是否在视场内

        Args:
            pts_rect: 在相机坐标系下的点云 (N, 3)
            img_shape: 图像的尺寸 [height, width]
            calib: 标定信息的实例

        Returns:
            pts_valid_flag: 有效点标志 (N,), True表示在视场内，False表示不在

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)

        # 判断投影点是否在图像范围内
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)

        # 深度 > 0, 才可以判断在fov视角
        # pts_valid_flag=array([ True,   True,  True, False,   True, True,.....])之类的，一共有M个 
        # 用于判断该点云能否有效 （是否用于训练）
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag



    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures
        # 线程函数
        def process_single_scene(sample_idx):
            # 打印样本索引
            print('%s sample_idx: %s' % (self.split, sample_idx))
            
            # 定义info空字典
            info = {}
            
            # 点云信息：点云特征维度和索引
            pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            
            # 添加点云信息
            info['point_cloud'] = pc_info
            
            # 图像信息：索引和图像高宽
            image_info = {'image_idx': sample_idx, 'image_shape': self.get_image_shape(sample_idx)}
            
            # 添加图像信息
            info['image'] = image_info

            # 根据索引获取Calibration对象
            calib = self.get_calib(sample_idx)
            
            # 构造相机矩阵P2
            P_left = np.concatenate([calib.P_left, np.array([[0., 0., 0., 1.]])], axis=0)
            
            # 构造R0_rect矩阵
            R0_4x4 = np.zeros([4, 4], dtype=calib.R0_rect.dtype)
            R0_4x4[3, 3] = 1.
            R0_4x4[:3, :3] = calib.R0_rect
            
            # 构造Tr_velo_to_cam矩阵
            V2C_4x4 = np.concatenate([calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
            
            # 标定信息：P2、R0_rect和T_V_C
            calib_info = {'P_left': P_left, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4}
            
            # 添加标定信息
            info['calib'] = calib_info

            if has_label:
                # 根据索引读取label，构造object列表
                obj_list = self.get_label(sample_idx)
                annotations = {}
                
                # 根据属性将所有obj_list的属性添加进annotations
                annotations['name'] = np.array([obj.cls_type for obj in obj_list])
                annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
                annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
                # annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
                annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)

                annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
                annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
                annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
                annotations['score'] = np.array([obj.score for obj in obj_list])
                annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

                # 计算有效物体的个数，如6个
                num_objects = len([obj.cls_type for obj in obj_list])
                # 总物体的个数 10个
                num_gt = len(annotations['name'])
                index = list(range(num_objects))
                # 由此可以得到 index=[0,1,2,3,4,5]
                annotations['index'] = np.array(index, dtype=np.int32)

                # 假设有效物体的个数是N
                # 取有效物体的 location（N,3）、dimensions（N,3）、rotation_y（N,1）信息
                loc = annotations['location'][:num_objects]
                dims = annotations['dimensions'][:num_objects]
                rots = annotations['rotation_y'][:num_objects]
                # 通过计算得到在lidar坐标系下的坐标，loc_lidar:（N,3）
                # 因为项目数据集本身就是在雷达坐标系下，因此不需要转换坐标系
                # loc_lidar = calib.rect_to_lidar(loc)
                loc_lidar = loc
                # 分别取 dims中的第一列、第二列、第三列：l,h,w（N,1）
                l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                
                # 因为waytuos的数据集本身就是在物体中心所以就没有这一步了
                # loc_lidar[:, 2] += h[:, 0] / 2 # 将物体的坐标原点由物体底部中心移到物体中心

                # (N, 7) [x, y, z, dx, dy, dz, heading] 
                # np.newaxis在列上增加一维，因为rots是(N,)
                # -(np.pi / 2 + rots[..., np.newaxis]) 应为在waytuos中，camera坐标系下定义物体朝向与camera的x轴夹角顺时针为正，逆时针为负
                # 在pcdet中，lidar坐标系下定义物体朝向与lidar的x轴夹角逆时针为正，顺时针为负，所以二者本身就正负相反
                # pi / 2是坐标系x轴相差的角度(如图所示)
                # camera:         lidar:
                # Y                    X
                # |                    |
                # |____X         Y_____|
                # gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
                gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, rots[..., np.newaxis]], axis=1)
                annotations['gt_boxes_lidar'] = gt_boxes_lidar

                if count_inside_pts:
                    # 根据索引获取点云
                    points = self.get_lidar(sample_idx)
                    
                    # 根据索引获取Calibration对象
                    calib = self.get_calib(sample_idx)
                    
                    # 将lidar坐标系的点变换到rect坐标系
                    pts_rect = calib.lidar_to_rect(points[:, 0:3])
                    
                    # 返回true or false list判断点云是否在fov下，判断该点云能否有效 （是否用于训练）
                    fov_flag = self.get_fov_flag(pts_rect, info['image']['image_shape'], calib)
                    
                    # 提取有效点
                    pts_fov = points[fov_flag]
                    
                    # gt_boxes_lidar是(N,7)  [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
                    # 返回值corners_lidar为（N,8,3）
                    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                    
                    # num_gt是这一帧图像里物体的总个数，假设为10，
                    # 则num_points_in_gt=array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1], dtype=int32)
                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                    # num_objects是有效物体的个数，为N，假设为N=6
                    for k in range(num_objects):
                        # in_hull函数是判断点云是否在bbox中，(是否在物体的2D检测框中)
                        # 如果是，返回flag
                        # 运用到了“三角剖分”的概念和方法
                        # 输入是当前帧FOV视角点云和第K个box信息
                        flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar[k])
                        
                        # 计算框内包含的点云
                        num_points_in_gt[k] = flag.sum()
                    
                    # 添加框内点云数量信息
                    annotations['num_points_in_gt'] = num_points_in_gt

                # 添加注释信息
                info['annos'] = annotations

            return info


        # 根据输入的sample_id_list或使用默认的self.sample_id_list，创建一个线程池
        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            # 使用executor.map并发地调用process_single_scene函数处理多帧数据
            # process_single_scene是一个用于处理单帧数据的函数，sample_id_list中的每个元素代表一帧数据的索引
            # executor.map返回一个迭代器，其中的每个元素都是对应帧的处理结果（字典）
            infos = executor.map(process_single_scene, sample_id_list)

        # 将迭代器转换为列表，得到所有帧的处理结果（字典）的列表
        # 每个元素代表了一帧的信息，包含点云、图像、标定信息以及可能的标签等
        return list(infos)


    # 用trainfile的groundtruth产生groundtruth_database，
    # 只保存训练数据中的gt_box及其包围的点的信息，用于数据增强
    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch
        # 如果是“train”，创建的路径是  /data/waytuos/gt_database
        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        # 在/data/waytuos/下创建保存waytuos_dbinfos_train的文件
        db_info_save_path = Path(self.root_path) / ('waytuos_dbinfos_%s.pkl' % split)
        # parents=True，可以同时创建多级目录
        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        # 传入的参数info_path是一个.pkl文件，ROOT_DIR/data/waytuos/waytuos_infos_train.pkl
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        # 读取infos里的每个info的信息，一个info是一帧的数据
        for k in range(len(infos)):
            # 输出的是 第几个样本 如7/780
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            # 取当前帧的信息 info
            info = infos[k]
            # 取里面的样本序列，其实就是data/waytuos/ImageSets/train.txt里面的数字序列
            # 如000000，000003,000007....
            sample_idx = info['point_cloud']['lidar_idx']
            # 读取该bin文件类型，并将点云数据以numpy的格式输出
            # points是一个数组（M,4）
            points = self.get_lidar(sample_idx)
            # 读取注释信息
            annos = info['annos']
            # name的数据是['Mid_size_car','Mid_size_car','Full_size_car'...]表示当前帧里面的所有物体objects
            names = annos['name']
            # difficulty：[0,1,2,-1,0,0,-1,1,...,]里面具体物体的难度，长度为总物体的个数
            difficulty = annos['difficulty']
            # bbox是一个数组，表示物体2D边框的个数，
            # 假设有效物体为N,则bbox:(N,4）
            bbox = annos['bbox']
            gt_boxes = annos['gt_boxes_lidar']

            # num_obj是有效物体的个数，为N
            num_obj = gt_boxes.shape[0]
            # 返回每个box中的点云索引[0 0 0 1 0 1 1...]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                # 创建文件名，并设置保存路径，最后文件如：000007_Cyclist_3.bin
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                # /data/waytuos/gt_database/000007_Cyclist_3.bin
                filepath = database_save_path / filename
                # point_indices[i] > 0得到的是一个[T,F,T,T,F...]之类的真假索引，共有M个
                # 再从points中取出相应为true的点云数据，放在gt_points中
                gt_points = points[point_indices[i] > 0]

                # 将第i个box内点转化为局部坐标
                gt_points[:, :3] -= gt_boxes[i, :3]
                # 把gt_points的信息写入文件里
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    # 获取文件相对路径
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    # 根据当前物体的信息组成info
                    db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                               'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i]}
                    # 把db_info信息添加到 all_db_infos字典里面
                    if names[i] in all_db_infos:
                        # 如果存在该类别则追加
                        all_db_infos[names[i]].append(db_info)
                    else:
                        # 如果不存在该类别则新增
                        all_db_infos[names[i]] = [db_info]
        # 输出数据集中不同类别物体的个数
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))
        # 把所有的all_db_infos写入到文件里面
        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict: 
                为字典包含batch的calib和image_shape等信息，通过frame_id索引
            pred_dicts: list of pred_dicts 预测列表包含:
                pred_boxes: (N, 7), Tensor 预测的框，包含七个信息
                pred_scores: (N), Tensor   预测得分
                pred_labels: (N), Tensor   预测的类比
            class_names:
            output_path:

        Returns:

        """
        # 获取预测后的模板字典pred_dict，全部定义为全零的向量
        # waytuos格式增加了score和boxes_lidar信息
        # 参数num_samples表示这一帧里面的物体个数
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            """
            接收模型预测的在统一坐标系下表示的3D检测框，并转回自己所需格式,生成一帧的预测字典
            Args:
                batch_index:batch的索引id
                box_dict:预测的结果，字典包含pred_scores、pred_boxes、pred_labels等信息
            """
            #pred_scores: (N), Tensor      预测得分，N是这一帧预测物体的个数
            #pred_boxes: (N, 7), Tensor    预测的框，包含七个信息
            #pred_labels: (N), Tensor      预测的标签
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            # 定义一个帧的空字典，用来存放来自预测的信息
            pred_dict = get_template_prediction(pred_scores.shape[0])
            # 如果没有物体，则返回空字典
            if pred_scores.shape[0] == 0:
                return pred_dict

            # 获取该帧的标定和图像尺寸信息
            calib = batch_dict['calib'][batch_index]
            image_shape = batch_dict['image_shape'][batch_index].cpu().numpy()
            # 将预测的box3D转化从lidar系转化到camera系
            pred_boxes_camera = box_utils.boxes3d_lidar_to_waytuos_camera(pred_boxes, calib)
            # 将camera系下的box3D信息转化为box2D信息
            pred_boxes_img = box_utils.boxes3d_waytuos_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=image_shape
            )
            # 向刚刚创建的全零字典中填充预测信息，类别名，角度等信息（waytuos格式）
            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            # pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            # pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            # pred_dict['location'] = pred_boxes_camera[:, 0:3]
            # pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            # 项目需要的本身就是雷达坐标系下的坐标因此不用变换
            pred_dict['dimensions'] = pred_boxes[:, 3:6]
            pred_dict['location'] = pred_boxes[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            # 返回预测字典
            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            # 获取id帧号
            frame_id = batch_dict['frame_id'][index]
            # 获取单帧的预测结果
            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                # 定义输出结果的文件路径比如： data/waytuos/output/000007.txt
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    # 从单帧预测dict中，提取bbox、loc和dims等信息
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> lwh
                    # 将预测信息输出至终端同时写入文件
                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][0], dims[idx][2], dims[idx][1], loc[idx][1],
                                 loc[idx][2], loc[idx][0], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos # 返回所有预测信息

    # def evaluation(self, det_annos, class_names, **kwargs):
    #     # 如果 'annos' 不在 waytuos_infos 的第一个元素中，直接返回空字典
    #     if 'annos' not in self.waytuos_infos[0].keys():
    #         return None, {}

    #     # 导入目标检测评估工具
    #     from .waytuos_object_eval_python import eval as waytuos_eval

    #     # 复制目标检测结果
    #     eval_det_annos = copy.deepcopy(det_annos)

    #     # 复制所有数据的真值标注
    #     eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.waytuos_infos]

    #     # 根据目标检测的真值和预测值，计算四个检测指标 分别为 bbox、bev、3d和 aos
    #     ap_result_str, ap_dict = waytuos_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)

    #     return ap_result_str, ap_dict


    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.waytuos_infos) * self.total_epochs
        #等于返回训练帧的总个数，等于图片的总个数，帧的总个数
        return len(self.waytuos_infos)

    # 将点云与3D标注框均转至前述统一坐标定义下，送入数据基类提供的self.prepare_data()
    def __getitem__(self, index):
        """
        从pkl文件中获取相应index的info，然后根据info['point_cloud']['lidar_idx']确定帧号，进行数据读取和其他info字段的读取
        初步读取的data_dict,要传入prepare_data（dataset.py父类中定义）进行统一处理，然后即可返回
        """
        # index = 4
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.waytuos_infos)

        # 取出第index帧的信息
        info = copy.deepcopy(self.waytuos_infos[index])

        # 获取采样的序列号，在train.txt文件里的数据序列号
        sample_idx = info['point_cloud']['lidar_idx']
        # 获取该序列号相应的 图像宽高
        img_shape = info['image']['image_shape']
        # 获取该序列号相应的相机参数，如P_left,R0_rect,V2C
        calib = self.get_calib(sample_idx)
        # 获取item列表
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        # 定义输入数据的字典包含帧id和标定信息
        input_dict = {
            'frame_id': sample_idx,
            'calib': calib,
        }

        if 'annos' in info:
            # 获取该帧信息中的 annos
            annos = info['annos']
            # 得到有效物体object(N个)的位置、大小和角度信息（N,3）,(N,3),(N)
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']

            # 构造camera系下的label（N,7），再转换到lidar系下
            # boxes3d_camera: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
            # boxes3d_lidar: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
            # gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            # gt_boxes_lidar = box_utils.boxes3d_waytuos_camera_to_lidar(gt_boxes_camera, calib)
            # 项目数据集本身就是雷达坐标系下，因此不需要转换成雷达坐标系了
            gt_boxes_lidar = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)

            # 将新的键值对 添加到输入的字典中去，此时输入中有四个键值对了
            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })

            # 如果get_item_list中有gt_boxes2d，则将bbox加入到input_dict中
            if "gt_boxes2d" in get_item_list:
                input_dict['gt_boxes2d'] = annos["bbox"]
            
            #如果有路面信息，则加入进去
            road_plane = self.get_road_plane(sample_idx)
            if road_plane is not None:
                input_dict['road_plane'] = road_plane

        # 加入点云，如果要求FOV视角，则对点云进行裁剪后加入input_dict
        if "points" in get_item_list:
            points = self.get_lidar(sample_idx)
            if self.dataset_cfg.FOV_POINTS_ONLY:
                pts_rect = calib.lidar_to_rect(points[:, 0:3])
                fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
                points = points[fov_flag]
            input_dict['points'] = points

        # 加入图片
        if "images" in get_item_list:
            input_dict['images'] = self.get_image(sample_idx)

        # 加入深度图
        if "depth_maps" in get_item_list:
            input_dict['depth_maps'] = self.get_depth_map(sample_idx)

        # 加入标定信息
        if "calib_matricies" in get_item_list:
            input_dict["trans_lidar_to_cam"], input_dict["trans_cam_to_img"] = calib_to_matricies(calib)

        # 将输入数据送入prepare_data进一步处理，形成训练数据
        data_dict = self.prepare_data(data_dict=input_dict)

        # 加入图片宽高信息
        data_dict['image_shape'] = img_shape
        return data_dict


def create_waytuos_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    """
    生成.pkl文件（对train/test/val均生成相应文件），提前读取点云格式、image格式、calib矩阵以及label
    """
    # 创建waytuosDataset对象，用于加载waytuos数据集
    dataset = Waytuos(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    # 定义训练和验证集的划分
    train_split, val_split = 'train', 'val'
    # 定义保存文件的路径和名称
    train_filename = save_path / ('waytuos_infos_%s.pkl' % train_split)  # /data/waytuos/waytuos_infos_train.pkl
    val_filename = save_path / ('waytuos_infos_%s.pkl' % val_split)  # /data/waytuos/waytuos_infos_val.pkl
    trainval_filename = save_path / 'waytuos_infos_trainval.pkl'
    test_filename = save_path / 'waytuos_infos_test.pkl'

    print('---------------Start to generate data infos---------------')

    # 针对训练集进行信息生成
    dataset.set_split(train_split) 
    # 执行完上一步，得到train相关的保存文件，以及sample_id_list的值为train.txt文件下的数字
    # 下面是得到train.txt中序列相关的所有点云数据的信息，并且进行保存
    waytuos_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(train_filename, 'wb') as f:
        pickle.dump(waytuos_infos_train, f)
    print('waytuos info train file is saved to %s' % train_filename)

    # 对验证集的数据进行信息统计并保存
    dataset.set_split(val_split)
    waytuos_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(val_filename, 'wb') as f:
        pickle.dump(waytuos_infos_val, f)
    print('waytuos info val file is saved to %s' % val_filename)

    # 将训练集和验证集的信息合并写到一个文件里
    with open(trainval_filename, 'wb') as f:
        pickle.dump(waytuos_infos_train + waytuos_infos_val, f)
    print('waytuos info trainval file is saved to %s' % trainval_filename)

    # 对测试集的数据进行信息统计并保存
    dataset.set_split('test')
    waytuos_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    with open(test_filename, 'wb') as f:
        pickle.dump(waytuos_infos_test, f)
    print('waytuos info test file is saved to %s' % test_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    # 用trainfile产生groundtruth_database
    # 只保存训练数据中的gt_box及其包围的点的信息，用于数据增强
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import sys

    import yaml
    from pathlib import Path
    from easydict import EasyDict
    # 从命令行参数中读取配置文件路径
    dataset_cfg = EasyDict(yaml.load(open('/root/autodl-fs/pointpillar/tools/cfgs/dataset_configs/waytuos_dataset.yaml'), Loader=yaml.FullLoader))
    # 获取OpenPCDet根目录
    ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
    # 调用create_waytuos_infos函数
    create_waytuos_infos(
        dataset_cfg=dataset_cfg,
        class_names=['Mid_size_car', 'Full_size_car', 'Pedestrian'],
        data_path=ROOT_DIR / 'data' / 'waytuos',  # OpenPCDet/data/waytuos
        save_path=ROOT_DIR / 'data' / 'waytuos'  # OpenPCDet/data/waytuos
    )
