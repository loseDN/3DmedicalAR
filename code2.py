import sys
import os
import glob
import pydicom
import numpy as np
import vtk
import SimpleITK as sitk
import imageio
import webbrowser
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QSlider, QLabel, QComboBox, QFileDialog, QMessageBox,
                             QStatusBar, QToolBar, QSpinBox, QTabWidget, QSplitter, QMenuBar, QShortcut)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from tqdm import tqdm


class AdvancedSlicerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("交互式3D医学影像与AR辅助诊断创意平台")
        self.setGeometry(100, 100, 1600, 1000)
        self.dcm_files = []
        self.volume_data = None
        self.pixel_spacing = None
        self.slice_thickness = None
        self.pixel_range = (0, 1000)
        self.current_slice = 0
        self.roi_actor = None
        self.init_ui()
        self.setup_vtk()
        self.apply_style("light")
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("准备就绪")
        self.setup_shortcuts()

    def init_ui(self):
        # 菜单栏
        menubar = QMenuBar()
        self.setMenuBar(menubar)
        file_menu = menubar.addMenu("文件")
        file_menu.addAction("加载DICOM", self.load_dicom_files)
        file_menu.addAction("AR预览", self.export_ar_preview)
        file_menu.addAction("退出", self.close)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 分割器布局
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # 左侧：3D和2D视图
        view_widget = QWidget()
        view_layout = QVBoxLayout(view_widget)
        splitter.addWidget(view_widget)

        # 3D视图
        self.vtk_widget = QVTKRenderWindowInteractor(view_widget)
        view_layout.addWidget(self.vtk_widget, stretch=3)

        # 2D切片视图
        self.slice_tabs = QTabWidget()
        self.axial_canvas = FigureCanvas(plt.Figure(figsize=(4, 4)))
        self.coronal_canvas = FigureCanvas(plt.Figure(figsize=(4, 4)))
        self.sagittal_canvas = FigureCanvas(plt.Figure(figsize=(4, 4)))
        self.slice_tabs.addTab(self.axial_canvas, "轴向")
        self.slice_tabs.addTab(self.coronal_canvas, "冠状")
        self.slice_tabs.addTab(self.sagittal_canvas, "矢状")
        view_layout.addWidget(self.slice_tabs, stretch=2)

        # 切片导航
        slice_nav = QHBoxLayout()
        self.slice_spinbox = QSpinBox()
        self.slice_spinbox.setRange(0, 0)
        self.slice_spinbox.valueChanged.connect(self.update_slice_view)
        slice_nav.addWidget(QLabel("切片："))
        slice_nav.addWidget(self.slice_spinbox)
        view_layout.addLayout(slice_nav)

        # 右侧：控制和分析面板
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        splitter.addWidget(control_widget)
        splitter.setSizes([1000, 600])

        # 工具栏
        toolbar = QToolBar("工具栏")
        self.addToolBar(toolbar)
        toolbar.addAction("加载DICOM", self.load_dicom_files)
        toolbar.addAction("导出图像", self.export_image)
        toolbar.addAction("导出视频", self.export_video)
        toolbar.addAction("导出STL", self.export_stl)
        toolbar.addAction("导出OBJ", self.export_obj)
        toolbar.addAction("AR预览", self.export_ar_preview)

        # 模块选择
        self.module_combo = QComboBox()
        self.module_combo.addItems(["可视化", "分割", "分析"])
        self.module_combo.currentTextChanged.connect(self.switch_module)
        control_layout.addWidget(QLabel("模块："))
        control_layout.addWidget(self.module_combo)

        # 可视化面板
        self.vis_panel = QWidget()
        vis_layout = QVBoxLayout(self.vis_panel)
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["浅色", "深色"])
        self.theme_combo.currentTextChanged.connect(self.toggle_theme)
        vis_layout.addWidget(QLabel("主题："))
        vis_layout.addWidget(self.theme_combo)
        self.render_combo = QComboBox()
        self.render_combo.addItems(["体视显微镜", "表面渲染", "光照体视"])
        self.render_combo.currentTextChanged.connect(self.update_rendering)
        vis_layout.addWidget(QLabel("渲染风格："))
        vis_layout.addWidget(self.render_combo)
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.valueChanged.connect(self.update_rendering)
        vis_layout.addWidget(QLabel("透明度："))
        vis_layout.addWidget(self.opacity_slider)

        # 分割面板
        self.seg_panel = QWidget()
        seg_layout = QVBoxLayout(self.seg_panel)
        self.roi_button = QPushButton("交互式ROI")
        self.roi_button.clicked.connect(self.start_roi_selection)
        seg_layout.addWidget(self.roi_button)
        self.ai_seg_button = QPushButton("AI分割（未启用）")
        self.ai_seg_button.clicked.connect(self.ai_segmentation)
        seg_layout.addWidget(self.ai_seg_button)

        # 分析面板
        self.anal_panel = QWidget()
        anal_layout = QVBoxLayout(self.anal_panel)
        self.measure_button = QPushButton("测量距离")
        self.measure_button.clicked.connect(self.measure_distance)
        anal_layout.addWidget(self.measure_button)
        self.analysis_label = QLabel("分析结果：\n体视：N/A\n平均强度：N/A\nROI体视：N/A")
        anal_layout.addWidget(self.analysis_label)
        self.hist_canvas = FigureCanvas(plt.Figure(figsize=(4, 4)))
        anal_layout.addWidget(self.hist_canvas)

        # 初始显示可视化面板
        self.control_stack = QTabWidget()
        self.control_stack.addTab(self.vis_panel, "可视化")
        self.control_stack.addTab(self.seg_panel, "分割")
        self.control_stack.addTab(self.anal_panel, "分析")
        self.control_stack.setVisible(True)
        control_layout.addWidget(self.control_stack)
        control_layout.addStretch()

    def setup_vtk(self):
        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)
        self.renderer.SetBackground(0.1, 0.2, 0.4)
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(0, 0, 500)
        camera.SetFocalPoint(0, 0, 0)

    def setup_shortcuts(self):
        QShortcut("Ctrl+N", self, self.load_dicom_files)
        QShortcut("Up", self, lambda: self.slice_spinbox.setValue(self.slice_spinbox.value() + 1))
        QShortcut("Down", self, lambda: self.slice_spinbox.setValue(self.slice_spinbox.value() - 1))

    def apply_style(self, theme):
        if theme == "深色":
            self.setStyleSheet("""
                QMainWindow, QWidget { background-color: #2b2b2b; color: #ffffff; font-family: Microsoft YaHei; }
                QPushButton { background-color: #4a4a4a; color: #ffffff; border: 1px solid #555555; padding: 5px; }
                QPushButton:hover { background-color: #5a5a5a; }
                QComboBox, QSlider, QSpinBox { background-color: #4a4a4a; color: #ffffff; }
                QLabel, QStatusBar { color: #ffffff; }
                QToolBar, QMenuBar { background-color: #3a3a3a; }
            """)
        else:
            self.setStyleSheet("""
                QMainWindow, QWidget { background-color: #f0f0f0; color: #000000; font-family: Microsoft YaHei; }
                QPushButton { background-color: #e0e0e0; color: #000000; border: 1px solid #aaaaaa; padding: 5px; }
                QPushButton:hover { background-color: #d0d0d0; }
                QComboBox, QSlider, QSpinBox { background-color: #e0e0e0; color: #000000; }
                QLabel, QStatusBar { color: #000000; }
                QToolBar, QMenuBar { background-color: #d0d0d0; }
            """)

    def toggle_theme(self, theme):
        self.apply_style(theme)
        self.statusBar().showMessage(f"主题切换为 {theme}")

    def switch_module(self, module):
        self.control_stack.setCurrentIndex({"可视化": 0, "分割": 1, "分析": 2}[module])
        self.statusBar().showMessage(f"切换到 {module} 模块")

    def load_dicom_files(self):
        default_dir = r"D:\rsna-2024-lumbar-spine-degenerative-classification\train_images\4003253\702807833"
        self.dcm_files = glob.glob(os.path.join(default_dir, "*.dcm"))
        if not self.dcm_files:
            files, _ = QFileDialog.getOpenFileNames(self, "选择DICOM文件", default_dir, "DICOM文件 (*.dcm)")
            self.dcm_files = files
        if not self.dcm_files:
            QMessageBox.warning(self, "警告", "未找到DICOM文件")
            self.statusBar().showMessage("未选择任何文件")
            return

        self.statusBar().showMessage(f"正在加载 {len(self.dcm_files)} 个DICOM文件...")
        try:
            self.process_dicom_files()
            self.create_3d_volume()
            self.update_rendering()
            self.update_slice_view()
            self.plot_histogram()
            self.analyze_volume()
            self.statusBar().showMessage(
                f"成功加载 {len(self.dcm_files)} 个DICOM文件，切片数：{self.volume_data.shape[0]}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理DICOM文件失败：{str(e)}")
            self.statusBar().showMessage("加载DICOM文件出错")

    def process_dicom_files(self):
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(
            r"D:\rsna-2024-lumbar-spine-degenerative-classification\train_images\4003253\702807833")
        if not series_ids:
            raise ValueError("未找到DICOM序列")

        max_series = max(series_ids, key=lambda sid: len(reader.GetGDCMSeriesFileNames(
            r"D:\rsna-2024-lumbar-spine-degenerative-classification\train_images\4003253\702807833",
            sid)))
        dicom_names = reader.GetGDCMSeriesFileNames(
            r"D:\rsna-2024-lumbar-spine-degenerative-classification\train_images\4003253\702807833",
            max_series)
        reader.SetFileNames(dicom_names)

        if len(dicom_names) < 2:
            raise ValueError(f"切片数（{len(dicom_names)}）不足以构建3D体视")

        image = reader.Execute()
        self.volume_data = sitk.GetArrayFromImage(image).astype(np.float32)
        self.pixel_spacing = image.GetSpacing()[:2]
        self.slice_thickness = image.GetSpacing()[2]
        self.pixel_range = (self.volume_data.min(), self.volume_data.max())
        num_slices = self.volume_data.shape[0]
        self.volume_data = np.clip(self.volume_data, self.pixel_range[0], self.pixel_range[1])

        print("Number of slices:", num_slices)
        print("Pixel value range:", self.pixel_range)
        print("Volume shape:", self.volume_data.shape)
        self.slice_spinbox.setRange(0, num_slices - 1)
        self.slice_spinbox.setValue(0)
        self.slice_spinbox.update()
        self.slice_tabs.update()

    def create_3d_volume(self):
        if self.volume_data is None:
            raise ValueError("体视数据未生成")
        image_data = vtk.vtkImageData()
        image_data.SetDimensions(self.volume_data.shape[2], self.volume_data.shape[1], self.volume_data.shape[0])
        image_data.SetSpacing(self.pixel_spacing[0], self.pixel_spacing[1], self.slice_thickness)

        flat_data = self.volume_data.ravel(order="F")
        vtk_array = vtk.vtkFloatArray()
        vtk_array.SetNumberOfTuples(flat_data.size)
        for i in range(flat_data.size):
            vtk_array.SetTuple1(i, flat_data[i])
        image_data.GetPointData().SetScalars(vtk_array)

        self.image_data = image_data
        print("3D volume created with dimensions:", self.volume_data.shape)

    def update_rendering(self):
        self.renderer.RemoveAllViewProps()
        if not hasattr(self, "image_data"):
            return

        min_val, max_val = self.pixel_range
        mid_val = (max_val + min_val) / 2

        if self.render_combo.currentText() == "体视显微镜":
            volume = vtk.vtkVolume()
            volume_mapper = vtk.vtkSmartVolumeMapper()
            volume_mapper.SetInputData(self.image_data)
            volume_mapper.SetRequestedRenderModeToDefault()

            color_func = vtk.vtkColorTransferFunction()
            color_func.AddRGBPoint(min_val, 0.0, 0.0, 0.0)
            color_func.AddRGBPoint(mid_val, 1.0, 0.8, 0.6)
            color_func.AddRGBPoint(max_val, 1.0, 1.0, 1.0)

            opacity_func = vtk.vtkPiecewiseFunction()
            opacity_value = self.opacity_slider.value() / 100.0
            opacity_func.AddPoint(min_val, 0.0)
            opacity_func.AddPoint(mid_val, opacity_value * 0.2)
            opacity_func.AddPoint(max_val, opacity_value)

            volume_property = vtk.vtkVolumeProperty()
            volume_property.SetColor(color_func)
            volume_property.SetScalarOpacity(opacity_func)
            volume_property.ShadeOn()
            volume_property.SetInterpolationTypeToLinear()

            volume.SetMapper(volume_mapper)
            volume.SetProperty(volume_property)
            self.renderer.AddVolume(volume)
        elif self.render_combo.currentText() == "光照体视":
            volume = vtk.vtkVolume()
            volume_mapper = vtk.vtkSmartVolumeMapper()
            volume_mapper.SetInputData(self.image_data)
            volume_mapper.SetRequestedRenderModeToDefault()

            color_func = vtk.vtkColorTransferFunction()
            color_func.AddRGBPoint(min_val, 0.0, 0.0, 0.0)
            color_func.AddRGBPoint(mid_val, 1.0, 0.8, 0.6)
            color_func.AddRGBPoint(max_val, 1.0, 1.0, 1.0)

            opacity_func = vtk.vtkPiecewiseFunction()
            opacity_value = self.opacity_slider.value() / 100.0
            opacity_func.AddPoint(min_val, 0.0)
            opacity_func.AddPoint(mid_val, opacity_value * 0.3)
            opacity_func.AddPoint(max_val, opacity_value * 0.8)

            gradient_opacity = vtk.vtkPiecewiseFunction()
            gradient_opacity.AddPoint(0, 0.0)
            gradient_opacity.AddPoint(100, 1.0)

            volume_property = vtk.vtkVolumeProperty()
            volume_property.SetColor(color_func)
            volume_property.SetScalarOpacity(opacity_func)
            volume_property.SetGradientOpacity(gradient_opacity)
            volume_property.ShadeOn()
            volume_property.SetInterpolationTypeToLinear()

            volume.SetMapper(volume_mapper)
            volume.SetProperty(volume_property)
            self.renderer.AddVolume(volume)
        else:
            contour = vtk.vtkMarchingCubes()
            contour.SetInputData(self.image_data)
            contour.SetValue(0, mid_val)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(contour.GetOutputPort())
            mapper.ScalarVisibilityOff()

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1.0, 0.8, 0.6)
            self.renderer.AddActor(actor)

        if self.roi_actor:
            self.renderer.AddActor(self.roi_actor)

        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()
        self.vtk_widget.update()
        print("Rendering updated")

    def update_slice_view(self):
        if not hasattr(self, "volume_data"):
            return
        self.current_slice = self.slice_spinbox.value()
        num_slices, rows, cols = self.volume_data.shape

        slice_data = self.volume_data[self.current_slice, :, :]
        slice_data = (slice_data - self.pixel_range[0]) / (self.pixel_range[1] - self.pixel_range[0])
        slice_data = np.clip(slice_data, 0, 1)

        axial_fig = self.axial_canvas.figure
        axial_fig.clear()
        ax = axial_fig.add_subplot(111)
        ax.imshow(slice_data, cmap="gray")
        ax.set_title(f"Axial slices {self.current_slice + 1}/{num_slices}")
        ax.axis("off")
        self.axial_canvas.draw()

        coronal_slice = self.volume_data[:, rows // 2, :]
        coronal_slice = (coronal_slice - self.pixel_range[0]) / (self.pixel_range[1] - self.pixel_range[0])
        coronal_slice = np.clip(coronal_slice, 0, 1)
        coronal_fig = self.coronal_canvas.figure
        coronal_fig.clear()
        ax = coronal_fig.add_subplot(111)
        ax.imshow(coronal_slice.T, cmap="gray", aspect=self.slice_thickness / self.pixel_spacing[1])
        ax.set_title("Coronal sections")
        ax.axis("off")
        self.coronal_canvas.draw()

        sagittal_slice = self.volume_data[:, :, cols // 2]
        sagittal_slice = (sagittal_slice - self.pixel_range[0]) / (self.pixel_range[1] - self.pixel_range[0])
        sagittal_slice = np.clip(sagittal_slice, 0, 1)
        sagittal_fig = self.sagittal_canvas.figure
        sagittal_fig.clear()
        ax = sagittal_fig.add_subplot(111)
        ax.imshow(sagittal_slice.T, cmap="gray", aspect=self.slice_thickness / self.pixel_spacing[0])
        ax.set_title("Sagittal slices")
        ax.axis("off")
        self.sagittal_canvas.draw()

    def plot_histogram(self):
        self.hist_canvas.figure.clear()
        if self.volume_data is not None:
            ax = self.hist_canvas.figure.add_subplot(111)
            flat_data = self.volume_data.flatten()
            ax.hist(flat_data, bins=100, color="blue", alpha=0.7)
            ax.set_title("Pixel intensity histogram")
            ax.set_xlabel("strength")
            ax.set_ylabel("frequency")
            self.hist_canvas.draw()

    def analyze_volume(self):
        if not hasattr(self, "volume_data"):
            return
        voxel_volume = self.pixel_spacing[0] * self.pixel_spacing[1] * self.slice_thickness / 1000  # mm³
        num_voxels = np.prod(self.volume_data.shape)
        total_volume = num_voxels * voxel_volume / 1000  # cm³
        mean_intensity = np.mean(self.volume_data)
        roi_volume = "N/A"
        if self.roi_actor:
            roi_volume = "计算中..."
        self.analysis_label.setText(
            f"分析结果：\n体视：{total_volume:.2f} cm³\n平均强度：{mean_intensity:.2f}\nROI体视：{roi_volume}"
        )

    def start_roi_selection(self):
        if not hasattr(self, "image_data"):
            QMessageBox.warning(self, "警告", "未加载数据，无法选择ROI")
            return
        contour = vtk.vtkMarchingCubes()
        contour.SetInputData(self.image_data)
        contour.SetValue(0, (self.pixel_range[0] + self.pixel_range[1]) / 2)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(contour.GetOutputPort())
        self.roi_actor = vtk.vtkActor()
        self.roi_actor.SetMapper(mapper)
        self.roi_actor.GetProperty().SetColor(0.0, 1.0, 0.0)
        self.update_rendering()
        self.statusBar().showMessage("ROI分割完成（基于阈值）")

    def ai_segmentation(self):
        QMessageBox.information(self, "提示",
                                "AI分割需预训练U-Net模型。请配置PyTorch和模型权重，参考：https://github.com/milesial/Pytorch-UNet")

    def measure_distance(self):
        QMessageBox.information(self, "提示", "距离测量功能开发中。请在3D视图中点击两点（需VTK交互逻辑）")

    def export_image(self):
        if not hasattr(self, "image_data"):
            QMessageBox.warning(self, "警告", "未加载数据，无法导出")
            return
        file_name, _ = QFileDialog.getSaveFileName(self, "保存图像", "", "PNG文件 (*.png);;JPEG文件 (*.jpg)")
        if file_name:
            window_to_image = vtk.vtkWindowToImageFilter()
            window_to_image.SetInput(self.vtk_widget.GetRenderWindow())
            window_to_image.Update()
            if file_name.endswith(".png"):
                writer = vtk.vtkPNGWriter()
            else:
                writer = vtk.vtkJPEGWriter()
            writer.SetFileName(file_name)
            writer.SetInputConnection(window_to_image.GetOutputPort())
            writer.Write()
            self.statusBar().showMessage("图像导出成功")

    def export_video(self):
        if not hasattr(self, "image_data"):
            QMessageBox.warning(self, "警告", "未加载数据，无法导出")
            return
        file_name, _ = QFileDialog.getSaveFileName(self, "保存视频", "", "MP4文件 (*.mp4)")
        if file_name:
            self.statusBar().showMessage("正在导出视频...")
            try:
                frames = []
                num_frames = 120
                for angle in tqdm(np.linspace(0, 360, num_frames), desc="生成视频帧"):
                    self.renderer.GetActiveCamera().Azimuth(360 / num_frames)
                    self.vtk_widget.GetRenderWindow().Render()
                    window_to_image = vtk.vtkWindowToImageFilter()
                    window_to_image.SetInput(self.vtk_widget.GetRenderWindow())
                    window_to_image.Update()
                    vtk_image = window_to_image.GetOutput()
                    width, height, _ = vtk_image.GetDimensions()
                    vtk_array = vtk_image.GetPointData().GetScalars()
                    components = vtk_array.GetNumberOfComponents()
                    arr = np.frombuffer(vtk_array, dtype=np.uint8).reshape(height, width, components)
                    frames.append(arr)
                with imageio.get_writer(file_name, fps=60) as writer:
                    for frame in frames:
                        writer.append_data(frame)
                self.statusBar().showMessage("视频导出成功")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出视频失败：{str(e)}")
                self.statusBar().showMessage("导出视频出错")

    def export_stl(self):
        if not hasattr(self, "image_data"):
            QMessageBox.warning(self, "警告", "未加载数据，无法导出")
            return
        file_name, _ = QFileDialog.getSaveFileName(self, "保存STL", "", "STL文件 (*.stl)")
        if file_name:
            contour = vtk.vtkMarchingCubes()
            contour.SetInputData(self.image_data)
            contour.SetValue(0, (self.pixel_range[0] + self.pixel_range[1]) / 2)
            writer = vtk.vtkSTLWriter()
            writer.SetFileName(file_name)
            writer.SetInputConnection(contour.GetOutputPort())
            writer.Write()
            self.statusBar().showMessage("STL导出成功")

    def export_obj(self, file_name=None):
        if not hasattr(self, "image_data"):
            QMessageBox.warning(self, "警告", "未加载数据，无法导出")
            return False
        if file_name is None:
            file_name, _ = QFileDialog.getSaveFileName(self, "保存OBJ", "", "OBJ文件 (*.obj)")
        if not file_name:
            return False
        contour = vtk.vtkMarchingCubes()
        contour.SetInputData(self.image_data)
        contour.SetValue(0, (self.pixel_range[0] + self.pixel_range[1]) / 2)
        # Scale down for AR compatibility (adjust to real-world size, e.g., mm to meters)
        transform = vtk.vtkTransform()
        transform.Scale(0.001, 0.001, 0.001)  # Convert mm to meters
        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetInputConnection(contour.GetOutputPort())
        transform_filter.SetTransform(transform)
        writer = vtk.vtkOBJWriter()
        writer.SetFileName(file_name)
        writer.SetInputConnection(transform_filter.GetOutputPort())
        writer.Write()
        self.statusBar().showMessage("OBJ导出成功（适配AR/VR）")
        return file_name

    def export_ar_preview(self):
        if not hasattr(self, "image_data"):
            QMessageBox.warning(self, "警告", "未加载数据，无法生成AR预览")
            return
        # Export OBJ file
        obj_dir = QFileDialog.getExistingDirectory(self, "选择AR预览输出目录")
        if not obj_dir:
            self.statusBar().showMessage("未选择输出目录")
            return
        obj_file = os.path.join(obj_dir, "model.obj")
        if not self.export_obj(obj_file):
            return

        # Generate HTML for WebXR-based AR
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>AR Medical Visualization</title>
            <script src="https://cdn.jsdelivr.net/npm/three@0.156.0/build/three.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/three@0.156.0/examples/js/loaders/OBJLoader.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/@ar-js-org/ar.js@2.2.2/three.js/build/ar.min.js"></script>
            <style>
                body {{ margin: 0; overflow: hidden; }}
                canvas {{ width: 100%; height: 100%; }}
            </style>
        </head>
        <body>
            <a-scene embedded arjs='sourceType: webcam; detectionMode: mono_and_matrix; matrixCodeType: 3x3;'>
                <a-marker preset='hiro'>
                    <a-entity>
                        <a-entity id="model" scale="0.1 0.1 0.1" position="0 0 0"></a-entity>
                    </a-entity>
                </a-marker>
                <a-entity camera></a-entity>
            </a-scene>
            <script>
                const scene = document.querySelector('a-scene');
                const modelEntity = document.getElementById('model');
                const loader = new THREE.OBJLoader();
                loader.load(
                    'model.obj',
                    function (object) {{
                        object.traverse(function (child) {{
                            if (child.isMesh) {{
                                child.material = new THREE.MeshStandardMaterial({{ color: 0x00ff00 }});
                            }}
                        }});
                        const threeScene = modelEntity.object3D;
                        threeScene.add(object);
                    }},
                    function (xhr) {{ console.log((xhr.loaded / xhr.total * 100) + '% loaded'); }},
                    function (error) {{ console.error('Error loading OBJ:', error); }}
                );
                const light = new THREE.AmbientLight(0xffffff, 0.5);
                scene.object3D.add(light);
                const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
                directionalLight.position.set(0, 1, 0);
                scene.object3D.add(directionalLight);
            </script>
        </body>
        </html>
        """
        html_file = os.path.join(obj_dir, "ar_preview.html")
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        # Open HTML in default browser
        webbrowser.open(f"file://{os.path.abspath(html_file)}")
        QMessageBox.information(self, "AR预览",
                                "AR预览已生成。请使用AR支持的设备（如智能手机）扫描Hiro标记以查看3D模型。访问 https://jeromeetienne.github.io/AR.js/data/images/Hiro.png 下载标记。")
        self.statusBar().showMessage("AR预览生成成功")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AdvancedSlicerApp()
    window.show()
    sys.exit(app.exec_())
