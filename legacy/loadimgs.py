"""
师姐提供的用于标注的工具, 可以修改亮度和对比度（修改前的）
"""
import sys
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QSlider, QPushButton, QHBoxLayout, QMainWindow, QFileDialog, QWidget, QShortcut, QScrollArea, QListWidget, QSplitter
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPixmap, QImage
from PIL import Image, ImageEnhance
from pathlib import Path
import glob

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
        self.brightness_slider = self.create_slider("Brightness", self.change_brightness)
        self.contrast_slider = self.create_slider("Contrast", self.change_contrast)
        self.zoom_slider = self.create_slider("Zoom", self.change_zoom)
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
        self.current_index = 0
        self.image = None
        self.image_display = None
        
        self.setup_shortcuts()

        self.show()


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


    def load_images_from_folder(self, folder_path):
        # List all image files in the folder
        folder_path = glob.glob(folder_path)
        self.image_list = [f for f in folder_path if f[-4:] in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']]
        self.image_list.sort()
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
            self.display_image()
        except Exception as e:
            print(f"Failed to load image: {e}")  # Print error if image fails to load

    def display_image(self):
        # Convert image to QPixmap and display it
        if self.image_display:
            image_qt = self.pil_to_qpixmap(self.image_display)
            if image_qt:
                self.image_label.setPixmap(image_qt)
            else:
                print("Failed to convert PIL image to QPixmap.")  # Debugging message
        else:
            print("No image loaded.")  # Debugging message

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
            self.current_index += 1
            self.load_image(self.image_list[self.current_index])

    def show_previous_image(self):
        # Show the previous image in the list
        if self.image_list and self.current_index > 0:
            self.current_index -= 1
            self.load_image(self.image_list[self.current_index])

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

    def change_brightness(self):
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

    def change_contrast(self):
        # Change the contrast of the image based on the slider value
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

    def change_zoom(self):
        # Change the zoom level based on the slider value
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





if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ImageViewer()

    # 直接写路径
    folder_path = "./cropped/*"
    viewer.load_images_from_folder(folder_path)

    sys.exit(app.exec_())
