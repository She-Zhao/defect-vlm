"""
将数据统一填充或裁剪成为640*640分辨率的图像
"""
# 对原始图像进行裁剪并保存，小于640*640的图像直接保存，640以上的进行裁剪
# 对bbox坐标进行变换，保存变换后的标注结果

# 对于MT NEU paint_ap，直接进行随机填充
# 对于engine al_primary al_final FXP，裁剪到640，640
# 对于kaggle-steel，沿宽度方向裁剪到640*640

# 以原始数据集的json文件作为输入
# 对图像进行遍历，同步更新并保存每个切图的bbox

class DataHandler:
    # 输入json文件，对数据进行分发，将图像和图像对应的bbox传递给ImageSlicer
    def __init__(self):
        pass


class ImageSlicer:
    # 输入一张原始图像和图像对应的bbox，对图像进行切分，并返回切分后图像的json文件。
    def __init__(self, target_size=640, ioa_threshold=0.7):
        """_summary_

        Args:
            target_size (int, optional): 切分后的图像大小. Defaults to 640.
            ioa_threshold (float, optional): _description_. Defaults to 0.7.
        """
        self.target_size = target_size
        self.ioa_threshold = ioa_threshold
        
    
    def slice_image(self, image):
        """对外接口，调用不同的切片函数"""
    
    def _random_padding(self, image):
        """应用于MT NEU paint_ap数据集，直接进行随机填充"""
        
    
    def _slice_from_width(self, image):
        """应用于kaggle-steel数据集，沿宽度方向裁剪到640*640"""
        
        
    def _slice_from_width_height(self, image):
        """engine、al_primary、al_final、FXP数据集，沿着宽度和高度两个方向裁剪到640，640""" 
    
    def _compute_ioa(self, bbox1, bbox2):
        """计算裁剪后的bbox / 原始bbox的比值"""
        
    
    def _update_bbox(self, image_coor):
        """找到当前图像切片中的bbox，并对其坐标进行更新"""
    