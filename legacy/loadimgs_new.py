"""
师姐提供的用于标注的工具, 可以修改亮度和对比度（修改后的）
"""
import sys
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout \
        , QSlider, QPushButton, QHBoxLayout, QMainWindow \
        , QWidget, QShortcut, QScrollArea, QListWidget \
        , QSplitter, QCheckBox, QDialog
from PyQt5.QtCore import Qt, QPoint, QRect
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
from pathlib import Path
import glob
import numpy as np
import os

CLASSES = ['breakage', 'inclusion', 'scratch', 'crater', 'run', 'bulge', 'condensate', 'pinhole']


class ClassSelectionDialog(QDialog):
    def __init__(self, classes, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Class")
        self.setMinimumSize(300, 400)
        self.classes = classes

        # 主布局
        layout = QVBoxLayout()

        # 添加提示信息
        label = QLabel("Select a class:")
        layout.addWidget(label)

        # 类别列表
        self.list_widget = QListWidget()
        for class_name in classes:
            self.list_widget.addItem(class_name)
        layout.addWidget(self.list_widget)

        # 按钮布局
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.cancel_button = QPushButton("Cancel")
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        # 信号与槽
        self.selected_class = None  # 用于存储用户选择
        self.list_widget.itemDoubleClicked.connect(self.on_item_double_clicked)
        self.ok_button.clicked.connect(self.on_ok_clicked)
        self.cancel_button.clicked.connect(self.on_cancel_clicked)

    def on_item_double_clicked(self, item):
        """双击列表项选择类别"""
        self.selected_class = item.text()
        self.accept()  # 关闭对话框并返回确认状态

    def on_ok_clicked(self):
        """点击 OK 按钮确认选择"""
        selected_item = self.list_widget.currentItem()
        if selected_item:
            self.selected_class = selected_item.text()
        self.accept()

    def on_cancel_clicked(self):
        """点击 Cancel 按钮取消操作"""
        self.selected_class = None
        self.reject()  # 关闭对话框并返回取消状态

    def get_selected_class(self):
        """获取用户选择的类别"""
        return self.classes.index(self.selected_class)


class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Viewer")
        self.setGeometry(100, 100, 800, 600)
        self.showMaximized()  # 窗口最大化
        
        # 滚动区域，用于显示图片
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.scroll_area.setWidget(self.image_label)

        # 右侧布局：操作面板
        self.controls_layout = QVBoxLayout()

        # 文件列表
        self.file_list_widget = QListWidget(self)
        self.file_list_widget.itemClicked.connect(self.on_file_selected)
        self.controls_layout.addWidget(self.file_list_widget)

        # 滑条：亮度、对比度、缩放
        self.brightness_slider = self.create_slider("Brightness", self.change_bcz)
        self.contrast_slider = self.create_slider("Contrast", self.change_bcz)
        self.zoom_slider = self.create_slider("Zoom", self.change_bcz)
        self.controls_layout.addWidget(QLabel("Brightness"))
        self.controls_layout.addWidget(self.brightness_slider)
        self.controls_layout.addWidget(QLabel("Contrast"))
        self.controls_layout.addWidget(self.contrast_slider)
        self.controls_layout.addWidget(QLabel("Zoom"))
        self.controls_layout.addWidget(self.zoom_slider)

        # 导航按钮
        self.prev_button = QPushButton("Previous", self)
        self.prev_button.clicked.connect(self.show_previous_image)
        self.next_button = QPushButton("Next", self)
        self.next_button.clicked.connect(self.show_next_image)
        self.controls_layout.addWidget(self.prev_button)
        self.controls_layout.addWidget(self.next_button)
        
        # 添加复选框
        self.show_original_checkbox = QCheckBox("Show Original", self)
        self.show_original_checkbox.stateChanged.connect(self.toggle_original_image)
        self.controls_layout.addWidget(self.show_original_checkbox)
        
        self.clear_rectangles_checkbox = QCheckBox("Clear All Rectangles")
        self.clear_rectangles_checkbox.setChecked(False)  # 默认不勾选
        self.clear_rectangles_checkbox.stateChanged.connect(self.toggle_rectangle_display)
        self.controls_layout.addWidget(self.clear_rectangles_checkbox)

        # 右侧操作面板容器
        self.controls_widget = QWidget()
        self.controls_widget.setLayout(self.controls_layout)

        # 使用 QSplitter 实现可调布局
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.scroll_area)  # 左侧图片显示区域
        self.splitter.addWidget(self.controls_widget)  # 右侧操作面板

        # 设置初始比例
        self.splitter.setStretchFactor(0, 80)  # 左侧占 3
        self.splitter.setStretchFactor(1, 1)  # 右侧占 1

        # 设置中心窗口布局
        widget = QWidget(self)
        layout = QHBoxLayout(widget)
        layout.addWidget(self.splitter)
        self.setCentralWidget(widget)

        # 初始化变量
        self.image_list = []
        self.label_path = None
        self.current_image_name = None
        self.current_index = 0
        self.image = None
        self.image_display = None
        self.show_rectangles = True
        
        self.rectangles = []  # 保存所有的矩形框，格式为 [(x1, y1, x2, y2)]
        self.current_rect = None  # 当前正在绘制的矩形框
        self.start_point = None  # 矩形框起点 (x, y)
        self.drawing_enabled = False
        self.selected_rectangle = None
        self.selected_x = None
        self.selected_y = None

        # self.dragging_rect = None  # 当前拖动的矩形框
        # self.drag_offset = QPoint()  # 记录拖动的偏移量
        # self.resizing_rect = None  # 当前调整大小的矩形框
            
        self.setup_shortcuts()

        self.show()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_W:  # 检测按下 'W' 键
            self.drawing_enabled = not self.drawing_enabled  # 切换绘制模式
            if self.drawing_enabled:
                # 设置鼠标指针为十字形
                self.setCursor(Qt.CrossCursor)
            else:
                # 恢复默认鼠标指针
                self.setCursor(Qt.ArrowCursor)
        elif event.key() == Qt.Key_F or event.key() == Qt.Key_Delete:  # 检测按下 'Delete' 键
            self.delete_selected_rectangle()  # 删除选中的矩形框
        elif event.key() == Qt.Key_S:  # 检测按下 'S' 键
            self.save_labels()
            
    def delete_selected_rectangle(self):
        if self.selected_rectangle is not None:
            # 根据索引删除选中的矩形框
            if 0 <= self.selected_rectangle < len(self.rectangles):
                deleted_rect = self.rectangles.pop(self.selected_rectangle)  # 从列表中移除
                print(f"Deleted rectangle: {deleted_rect}")  # Debugging message
            self.selected_rectangle = None  # 清除选中状态
            self.display_image()  # 更新图像显示
   
    def on_file_selected(self, item):
        # 获取文件名并加载对应图片
        selected_file = item.text()
        for file_path in self.image_list:
            if Path(file_path).name == selected_file:
                self.current_index = self.image_list.index(file_path)
                self.load_image(file_path)
                break

    def setup_shortcuts(self):
        # Shortcut for 'd' to go to next image
        self.shortcut_next = QShortcut(Qt.Key_D, self)
        self.shortcut_next.activated.connect(self.show_next_image)

        # Shortcut for 'a' to go to previous image
        self.shortcut_prev = QShortcut(Qt.Key_A, self)
        self.shortcut_prev.activated.connect(self.show_previous_image)

    def load_images_from_folder(self, folder_path, label_path):
        # List all image files in the folder
        folder_path = glob.glob(folder_path)
        self.image_list = [f for f in folder_path if f[-4:] in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']]
        self.image_list.sort()
        self.label_path = label_path
        # folder_path = Path(folder_path)  # 使用 Pathlib 来处理路径
        # self.image_list = [f for f in folder_path.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']]
        self.current_index = 0
        print(f"Found {len(self.image_list)} images in the folder.")  # Print number of images found
        
        self.file_list_widget.clear()  # 清空之前的列表
        for file_path in self.image_list:
            file_name = Path(file_path).name
            self.file_list_widget.addItem(file_name)
        
        if self.image_list:
            self.load_image(self.image_list[self.current_index])

    def load_image(self, image_path):
        # Load and display the image
        try:
            print(f"Loading image: {image_path}")  # Print the image path to debug
            self.image = Image.open(image_path)
            self.image = self.image.resize(
                (int(1200), int(1200)),
                Image.Resampling.LANCZOS  # Use LANCZOS for high-quality resizing
            )
            self.image_display = self.image
            self.setWindowTitle(f"Image Viewer - {Path(image_path).name}")
            print(f"Image loaded: {self.image.size} pixels")  # Print image size
            
            self.current_image_name = image_path.split('\\')[-1]
            txt_path = os.path.join(self.label_path, self.current_image_name[0:-4]+'.txt')
            self.rectangles = []  # 清空当前图片的矩形框列表
            if os.path.exists(txt_path):
                rects_cxywh = np.loadtxt(txt_path, ndmin=2)
                for i in range(rects_cxywh.shape[0]):
                    self.rectangles.append(self.cxywh2cxyxy(rects_cxywh[i]))
            self.change_bcz()
        except Exception as e:
            print(f"Failed to load image: {e}")  # Print error if image fails to load

    def cxywh2cxyxy(self, rect_cxywh):
        c, x, y, w, h = rect_cxywh
        x0 = x - w/2
        y0 = y - h/2
        x1 = x + w/2
        y1 = y + h/2
        rect_cxyxy = np.array([c, x0, y0, x1, y1])
        return rect_cxyxy
    
    def cxyxy2cxywh(self, rect_cxyxy):
        c, x0, y0, x1, y1 = rect_cxyxy
        x = (x0 + x1) / 2
        y = (y0 + y1) / 2
        w = x1 - x0
        h = y1 - y0
        rect_cxywh = np.array([c, x, y, w, h])
        return  rect_cxywh
    
    def pil_to_qpixmap(self, pil_image):
        # Convert PIL image to QPixmap for display
        try:
            pil_image = pil_image.convert("RGB")  # Ensure the image is in RGB mode
            data = pil_image.tobytes("raw", "RGB")  # Get raw pixel data
            qimage = QImage(data, pil_image.width, pil_image.height, QImage.Format_RGB888)  # Create QImage
            return QPixmap(qimage)  # Convert QImage to QPixmap
        except Exception as e:
            print(f"Error in converting PIL to QPixmap: {e}")  # Debugging message
            return None

    def show_next_image(self):
        # Show the next image in the list
        if self.image_list and self.current_index < len(self.image_list) - 1:
            self.save_labels()
            self.current_index += 1
            self.load_image(self.image_list[self.current_index])

    def show_previous_image(self):
        # Show the previous image in the list
        if self.image_list and self.current_index > 0:
            self.save_labels()
            self.current_index -= 1
            self.load_image(self.image_list[self.current_index])

    def save_labels(self):
        txt_path = os.path.join(self.label_path, self.current_image_name[0:-4]+'.txt')
        with open(txt_path, 'w') as file:
            for rect_cxyxy in self.rectangles:
                rect_cxywh = self.cxyxy2cxywh(rect_cxyxy)
                # 第一个数字保存为int，其余数字保留六位小数
                line = f"{int(rect_cxywh[0])} " + " ".join([f"{x:.6f}" for x in rect_cxywh[1:]])
                file.write(line + '\n')
                # print(line + '\n')

    def create_slider(self, label, callback):
        # Create a slider for brightness/contrast adjustment
        slider = QSlider(Qt.Horizontal)
        if label == 'Zoom':
            slider.setRange(1, 30)
            slider.setValue(10)
            slider.valueChanged.connect(callback)
        else:
            slider.setRange(0, 100)
            slider.setValue(20)
            slider.valueChanged.connect(callback)

        return slider

    def change_bcz(self):
        # Change the brightness of the image based on the slider value
        if self.image:
            self.image_display = self.image.copy()
            enhancer = ImageEnhance.Brightness(self.image_display)
            self.image_display = enhancer.enhance(self.brightness_slider.value() / 10)
            enhancer = ImageEnhance.Contrast(self.image_display)
            self.image_display = enhancer.enhance(self.contrast_slider.value() / 10)
            zoom_factor = self.zoom_slider.value() / 10  # Get zoom percentage (0.1 to 2.0)
            self.image_display = self.image_display.resize(
                (int(self.image.width * zoom_factor), int(self.image.height * zoom_factor)),
                Image.Resampling.LANCZOS  # Use LANCZOS for high-quality resizing
            )
            self.display_image()
            
    def toggle_original_image(self, state):
        if state == Qt.Checked:
            # 显示原图
            self.image_display = self.image.copy()  # 确保显示原始图像
            zoom_factor = self.zoom_slider.value() / 10  # Get zoom percentage (0.1 to 2.0)
            self.image_display = self.image_display.resize(
                (int(self.image.width * zoom_factor), int(self.image.height * zoom_factor)),
                Image.Resampling.LANCZOS  # Use LANCZOS for high-quality resizing
            )
        else:
            # 显示处理后的图像
            self.change_bcz()  # 更新为当前亮度
        
        self.display_image()            
          
    def toggle_rectangle_display(self):
        # 切换显示矩形框的状态
        self.show_rectangles = not self.clear_rectangles_checkbox.isChecked()
        self.display_image()
            
    def display_image(self):
        """
        显示图片，并绘制已记录的矩形框。
        """
        if self.image_display:
            # 在 PIL 图像上绘制矩形框
            image_to_draw = self.image_display.copy()
            if image_to_draw.mode == "L":  # PIL "L" 模式表示灰度图
                image_to_draw = image_to_draw.convert("RGB")
            draw = ImageDraw.Draw(image_to_draw)
            
            if self.show_rectangles:
                # 绘制已完成的矩形框
                for idx, rect in enumerate(self.rectangles):
                    whwh = (self.image_display.width, self.image_display.height, 
                            self.image_display.width, self.image_display.height)
                    rect2draw = tuple(rect[1:5] * whwh)
                    c = CLASSES[int(rect[0])]
                    # 显示类别标签
                    zoom_factor = self.zoom_slider.value() / 10
                    text_position = (rect2draw[0] + 0 * zoom_factor, rect2draw[1] - 55 * zoom_factor)  # 类别标签的位置
                    font = ImageFont.truetype("arial.ttf", 50 * zoom_factor)
                    draw.text(text_position, c, fill="red", font=font)  # 绘制类别标签
                    if self.selected_rectangle is not None and idx == self.selected_rectangle:
                        draw.rectangle(rect2draw, outline="red", width=4)
                    else:
                        draw.rectangle(rect2draw, outline="red", width=2)
                    
            # 绘制当前正在绘制的矩形框
            if self.current_rect is not None:#x0y0x1y1
                    whwh = (self.image_display.width, self.image_display.height, 
                            self.image_display.width, self.image_display.height)
                    rect2draw = tuple(self.current_rect * whwh)
                    draw.rectangle(rect2draw, outline="red", width=2)
                
            # 转换为 QPixmap 显示
            image_qt = self.pil_to_qpixmap(image_to_draw)
            if image_qt:
                self.image_label.setPixmap(image_qt)
            else:
                print("Failed to convert PIL image to QPixmap.")  # 调试信息
        else:
            print("No image loaded.")  # 调试信息(640, 312, 1430, 383)(666, 959, 1490, 1090)

    def mousePressEvent(self, event):
        """
        鼠标按下事件，用于记录矩形框的起点。
        """
        if self.drawing_enabled:
            if event.button() == Qt.LeftButton and self.image_display:
                # 转换 QLabel 坐标到图片像素坐标
                self.start_point = self.label_to_image(event.pos())
                self.current_rect = None
        else:
            # 转换为图片坐标
            self.selected_x, self.selected_y = self.label_to_image(event.pos())
            image_x, image_y = self.selected_x, self.selected_y
            if image_x is None or image_y is None:
                # 鼠标点击位置不在图片范围内
                self.selected_rectangle = None
                self.display_image()
                return

            # 检查点击位置是否在任何矩形框内
            for idx, rect in enumerate(self.rectangles):
                x_min, y_min, x_max, y_max = rect[1:5]

                if x_min <= image_x <= x_max and y_min <= image_y <= y_max:
                    # 鼠标点击在矩形框内
                    self.selected_rectangle = idx
                    self.display_image()
                    return

            # 鼠标未点击任何矩形框，取消选中
            self.selected_rectangle = None
            self.display_image()

    def mouseMoveEvent(self, event):
        """
        鼠标移动事件，用于实时更新当前矩形框。
        """
        if self.drawing_enabled:
            if self.start_point and self.image_display:
                # 转换 QLabel 坐标到图片像素坐标
                end_point = self.label_to_image(event.pos())
                x1, y1 = self.start_point
                x2, y2 = end_point
                self.current_rect = np.array([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
                self.display_image()  # 更新显示
        else:
            image_x, image_y = self.label_to_image(event.pos())
            move_x, move_y = (image_x - self.selected_x, image_y - self.selected_y)
            self.selected_x, self.selected_y = (image_x, image_y)
            if self.selected_rectangle is not None:
                temp_xyxy = self.rectangles[self.selected_rectangle][1:5] + [move_x, move_y, move_x, move_y]
                if temp_xyxy[0] >= 0 and temp_xyxy[2] <= 1:
                    self.rectangles[self.selected_rectangle][1] = temp_xyxy[0]
                    self.rectangles[self.selected_rectangle][3] = temp_xyxy[2]
                if temp_xyxy[1] >= 0 and temp_xyxy[3] <= 1:
                    self.rectangles[self.selected_rectangle][2] = temp_xyxy[1]
                    self.rectangles[self.selected_rectangle][4] = temp_xyxy[3]
                # print(temp_xyxy)
                self.display_image()
            
    def mouseReleaseEvent(self, event):
        """
        鼠标释放事件，用于记录完成的矩形框。
        """
        if self.drawing_enabled:
            if event.button() == Qt.LeftButton and self.current_rect is not None:
                temp = []
                dialog = ClassSelectionDialog(CLASSES, self)
                if dialog.exec_() == QDialog.Accepted:
                    selected_class = dialog.get_selected_class()
                    self.rectangles.append(np.concatenate(([selected_class], self.current_rect)))
                else:
                    print("No class selected.")
                
                self.current_rect = None
                self.start_point = None
                self.display_image()  # 更新显示
                self.drawing_enabled = False
                self.setCursor(Qt.ArrowCursor)

    def label_to_image(self, point):
        """
        将 QLabel 的坐标转换为图片像素坐标。
        """
        if not self.image_display or not self.image_label.pixmap():
            return (0, 0)
        
        label_width = self.image_label.width()
        label_height = self.image_label.height()
        image_width = self.image_display.width
        image_height = self.image_display.height
        x_res = int((label_width - image_width) / 2)
        y_res = int((label_height - image_height) / 2)
        
        # QLabel 坐标到图片像素坐标
        x = int(point.x() - x_res - 18) / image_width
        y = int(point.y() - y_res - 20) / image_height
        x = max(0, min(1, x))
        y = max(0, min(1, y))
        return (x, y)






if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ImageViewer()

    # 直接写路径
    folder_path = "D:/Project/Defect_Dataset/PaintDatasets24/Camera_single/stripe16_photo/cropped_gc0/*"
    label_path = "D:/Project/Defect_Dataset/PaintDatasets24/Camera_single/train_labels"
    viewer.load_images_from_folder(folder_path, label_path)

    sys.exit(app.exec_())
