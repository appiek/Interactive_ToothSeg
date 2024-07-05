# -*- coding: utf-8 -*-
import os

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QColor, QIcon
from PyQt5.QtWidgets import QComboBox, QHBoxLayout, QFileDialog, QWidget

from PyQt5 import QtCore, QtGui, QtWidgets
from medpy.io import load

import vtkmodules.all as vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from Tooth_Alveolar_Construction import Step1_CBCT_ToothAlveolar_Seg

error = vtk.vtkOutputWindow()
error.SetGlobalWarningDisplay(0)  # close error

color_data = [
    ("", "maxilla:"), ("#ae5170", "18"), ("#f3722c", "17"), ("#f8d62e", "16"), ("#f9c74f", "15"), ("#f29676", "14"),
    ("#ffa4c7", "13"), ("#f15bb5", "12"), ("#ab8dff", "11"),
    ("#965df5", "21"), ("#3a86ff", "22"), ("#00f9f9", "23"), ("#92c7a5", "24"), ("#c08552", "25"), ("#742c05", "26"),
    ("#a566a6", "27"), ("#00fa85", "28"),
    ("", "mandible:"), ("#ffbabd", "38"), ("#ffdaf5", "37"), ("#fbe02f", "36"), ("#aecefd", "35"), ("#8fdbf8", "34"),
    ("#f2ffcf", "33"), ("#efd758", "32"), ("#fadcc8", "31"),
    ("#ffc52f", "41"), ("#ff00a1", "42"), ("#aad6ed", "43"), ("#a5f8df", "44"), ("#73abff", "45"), ("#a799ff", "46"),
    ("#add3aa", "47"), ("#fdd8ce", "48"),
    ("", "supernumerary teeth:"), ("#78adfc", "51"), ("#cefb5c", "52"), ("#d1a386", "53"), ("#51faf0", "54"),
    ("#f09ea4", "55"), ("#247bfd", "56"), ("#7b67f9", "57"),
    ("#646464", "58"), ("#9297c2", "59"), ("#8f00a7", "60")
]
current_color = "#ae5170"
current_color_id = "18"
current_color_index = 1
actors = []
undo_stack = []
redo_stack = []
color_index_list = []
subject_name = ""


def setSubjectName(name):
    global subject_name
    subject_name = name


def getSubjectName():
    global subject_name
    return subject_name


def setColorIndexList(index):
    global color_index_list
    if index not in color_index_list:
        color_index_list.append(index)


def getColorIndexList():
    global color_index_list
    return color_index_list


def clearColorIndexList():
    global color_index_list
    color_index_list = []


def setColorIndex(index):
    global current_color_index
    current_color_index = index


def getColorIndex():
    global current_color_index
    return current_color_index


def setColor(color):
    global current_color
    current_color = color


def getColor():
    global current_color
    return current_color


def setColorId(id):
    global current_color_id
    current_color_id = id


def getColorId():
    global current_color_id
    return current_color_id


def setPaintActor(actor):
    global actors
    actors.append(actor)


def getPaintActors():
    global actors
    return actors


def clearPaintActors():
    global actors
    actors = []


def setUndoStack(stack):
    global undo_stack
    undo_stack.append(stack)


def getUndoStack():
    global undo_stack
    return undo_stack


def clearUndoStack():
    global undo_stack
    undo_stack = []


def setRedoStack(stack):
    global redo_stack
    redo_stack.append(stack)


def getRedoStack():
    global redo_stack
    return redo_stack


def LevelAndWidth(self):
    scalarRange = self.reader.GetOutput().GetScalarRange()
    window = (scalarRange[1] - scalarRange[0])
    level = (scalarRange[0] + scalarRange[1]) / -5.0
    return window, level


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return r, g, b


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        self.datapath = ''
        self.landmarkpath = None
        self.flag_seg = 'True'
        self.threshold_th = 0.95
        self.threshold_al = 0.9
        self.smoothfactor = 3.0
        self.erosion_radius_up = 3
        self.erosion_radius_low = 3
        self.output_file_path = './output/'
        self.gl_fileIsEmpty = True
        self.sliceXY = 0
        MainWindow.setWindowTitle('MainWindow')

        color = QtGui.QColor(255, 255, 255)
        MainWindow.setStyleSheet(f"background-color: {color.name()};")
        MainWindow.resize(1240, 920)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.widget = QtWidgets.QWidget(self.centralwidget)

        self.font = QtGui.QFont()
        self.font.setFamily("Times New Roman")
        self.font.setPointSize(12)

        self.font2 = QtGui.QFont()
        self.font2.setFamily("Times New Roman")
        self.font2.setPointSize(14)

        # ------------------ system layout ------------------------------
        self.system_layout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.system_layout.setSpacing(6)
        self.system_layout.addWidget(self.widget)

        # ---------------- view layout -------------------------------
        self.view_layout = QtWidgets.QVBoxLayout()
        self.view_layout.setSpacing(6)

        # ------------------------- XY Window ----------------------------
        self.frame_XY = QtWidgets.QFrame()
        self.frame_XY.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_XY.setFrameShadow(QtWidgets.QFrame.Raised)
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame_XY)
        self.pathDicomDir = global_dicomDir_path
        self.reader = vtk.vtkDICOMImageReader()
        self.reader.SetDirectoryName(self.pathDicomDir)
        self.reader.Update()
        self.viewer_XY = vtk.vtkImageViewer2()
        self.viewer_XY.SetInputData(self.reader.GetOutput())
        self.viewer_XY.SetupInteractor(self.vtkWidget)
        self.viewer_XY.SetRenderWindow(self.vtkWidget.GetRenderWindow())
        self.viewer_XY.SetSliceOrientationToXY()
        self.viewer_XY.Render()

        # ------------------XY Slider---------------------
        self.verticalSlider_XY = QtWidgets.QSlider()
        self.verticalSlider_XY.setOrientation(QtCore.Qt.Vertical)

        self.window_slider_layout = QtWidgets.QHBoxLayout()
        self.window_slider_layout.setSpacing(6)
        self.window_slider_layout.addWidget(self.vtkWidget)
        self.window_slider_layout.addWidget(self.verticalSlider_XY)

        # --------------------label------------------------、
        self.label_XY = QtWidgets.QLabel()
        self.label_XY.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.label_XY.setFont(self.font)

        self.view_layout.addLayout(self.window_slider_layout)
        self.view_layout.addWidget(self.label_XY)
        self.system_layout.addLayout(self.view_layout, 7)

        # -----------------------------toolbar layout---------------------------------------------------------------
        self.tool_bar_layout = QtWidgets.QVBoxLayout()
        self.tool_bar_layout.setSpacing(6)
        self.tool_bar_layout.setAlignment(Qt.AlignTop)
        # -----------------------------widget_contrast------------------------------------------------------------------
        self.widget_contrast = QtWidgets.QWidget(self.widget)
        self.widget_contrast.setFixedHeight(150)
        # self.widget_contrast.setMinimumSize(QtCore.QSize(200, 150))
        # self.widget_contrast.setMaximumSize(QtCore.QSize(240, 150))
        self.widget_contrast.setStyleSheet('''background-color: #fafafa''')
        self.contrast_vertical_layout = QtWidgets.QVBoxLayout(self.widget_contrast)
        self.contrast_vertical_layout.setContentsMargins(11, 11, 11, 11)
        self.contrast_vertical_layout.setSpacing(5)
        # -----------------------------title---------------------------------------
        self.title = QtWidgets.QLabel(self.widget_contrast)
        self.title.setFixedHeight(30)
        self.title.setStyleSheet("color:green")
        self.title.setFont(self.font2)
        self.contrast_vertical_layout.addWidget(self.title, Qt.AlignLeft | Qt.AlignTop)
        # ----------------------------window_level_slider--------------------------------------------------
        self.window_level_slider = QtWidgets.QSlider(self.widget_contrast)
        self.window_level_slider.setOrientation(QtCore.Qt.Horizontal)
        self.window_level_slider.setMaximum(3000)
        self.window_level_slider.setMinimum(-2000)
        self.window_level_slider.setSingleStep(1)
        self.window_level_slider.valueChanged.connect(self.window_level_slider_valuechange)
        self.contrast_vertical_layout.addWidget(self.window_level_slider, Qt.AlignLeft)
        self.window_level = QtWidgets.QLabel(self.widget_contrast)
        self.window_level.setFont(self.font)
        self.contrast_vertical_layout.addWidget(self.window_level, Qt.AlignCenter)
        # ---------------------------window_width_slider----------------------------------------------------
        self.window_width_slider = QtWidgets.QSlider(self.widget_contrast)
        self.window_width_slider.setOrientation(QtCore.Qt.Horizontal)
        self.window_width_slider.setMaximum(8000)
        self.window_width_slider.setMinimum(-2000)
        self.window_width_slider.setSingleStep(1)
        self.window_width_slider.valueChanged.connect(self.window_width_slider_valuechange)
        self.contrast_vertical_layout.addWidget(self.window_width_slider)
        self.window_width = QtWidgets.QLabel(self.widget_contrast)
        self.window_width.setFont(self.font)
        self.contrast_vertical_layout.addWidget(self.window_width, Qt.AlignCenter)
        self.tool_bar_layout.addWidget(self.widget_contrast)

        # --------------------- Labels Layout -------------------------------------
        self.widget_labels = QtWidgets.QWidget(self.widget)
        self.widget_labels.setFixedHeight(400)
        self.widget_labels.setStyleSheet('''background-color: #fafafa''')
        self.labels_vertical_layout = QtWidgets.QVBoxLayout(self.widget_labels)
        self.labels_vertical_layout.setContentsMargins(11, 11, 11, 11)
        self.labels_vertical_layout.setSpacing(5)
        self.labels_vertical_layout.setAlignment(Qt.AlignTop)
        # -----------------------------title---------------------------------------
        self.widget_title = QtWidgets.QLabel(self.widget_labels)
        self.widget_title.setMinimumSize(QtCore.QSize(200, 20))
        self.widget_title.setMaximumSize(QtCore.QSize(240, 20))
        self.widget_title.setStyleSheet("color:green")
        self.widget_title.setFont(self.font2)
        self.labels_vertical_layout.addWidget(self.widget_title, Qt.AlignLeft | Qt.AlignTop)

        self.pushButton_paint = QtWidgets.QPushButton(self.widget_labels)
        self.pushButton_paint.setFont(self.font)
        self.pushButton_paint.setCheckable(True)
        self.pushButton_paint.setAutoExclusive(False)
        self.pushButton_paint.clicked.connect(self.paint)
        self.paint_enable = False
        self.pushButton_clear = QtWidgets.QPushButton(self.widget_labels)
        self.pushButton_clear.setFont(self.font)
        self.pushButton_clear.setAutoExclusive(False)
        self.pushButton_clear.clicked.connect(self.clear)
        self.pushButton_undo = QtWidgets.QPushButton(self.widget_labels)
        self.pushButton_undo.setFont(self.font)
        self.pushButton_undo.setAutoExclusive(False)
        self.pushButton_undo.clicked.connect(self.undo)
        self.pushButton_redo = QtWidgets.QPushButton(self.widget_labels)
        self.pushButton_redo.setFont(self.font)
        self.pushButton_redo.setAutoExclusive(False)
        self.pushButton_redo.clicked.connect(self.redo)

        self.pushButton_layout = QtWidgets.QHBoxLayout()
        self.pushButton_layout.setSpacing(10)
        self.pushButton_layout.setContentsMargins(11, 11, 11, 11)
        self.pushButton_left_layout = QtWidgets.QVBoxLayout()
        self.pushButton_left_layout.setSpacing(5)
        self.pushButton_right_layout = QtWidgets.QVBoxLayout()
        self.pushButton_right_layout.setSpacing(5)
        self.pushButton_left_layout.addWidget(self.pushButton_paint)
        self.pushButton_left_layout.addWidget(self.pushButton_undo)
        self.pushButton_right_layout.addWidget(self.pushButton_clear)
        self.pushButton_right_layout.addWidget(self.pushButton_redo)
        self.pushButton_layout.addLayout(self.pushButton_left_layout)
        self.pushButton_layout.addLayout(self.pushButton_right_layout)
        self.labels_vertical_layout.addLayout(self.pushButton_layout)

        self.labels_title = QtWidgets.QLabel(self.widget_labels)
        self.labels_title.setFont(self.font)
        self.color_combobox = QComboBox()
        self.color_combobox.setFont(self.font)
        self.color_combobox.setMinimumSize(QtCore.QSize(180, 20))
        self.color_combobox.setMaximumSize(QtCore.QSize(200, 20))
        self.color_combobox.currentIndexChanged.connect(self.update_current_color)

        for color, name in color_data:
            if color == "":
                self.color_combobox.setFont(self.font)
                self.color_combobox.addItem(name)
            else:
                pix_color = QPixmap(20, 20)
                pix_color.fill(QColor(color))
                self.color_combobox.addItem(QIcon(pix_color), name)
        self.color_combobox.setCurrentIndex(1)

        self.label_color_layout = QHBoxLayout()
        self.label_color_layout.setAlignment(Qt.AlignLeft)
        self.label_color_layout.addWidget(self.labels_title)
        self.label_color_layout.addWidget(self.color_combobox)
        self.labels_vertical_layout.addLayout(self.label_color_layout)

        self.pushButton_load = QtWidgets.QPushButton(self.widget_labels)
        self.pushButton_load.setFont(self.font)
        self.pushButton_load.setAutoExclusive(False)
        self.pushButton_load.clicked.connect(self.load)

        self.pushButton_save = QtWidgets.QPushButton(self.widget_labels)
        self.pushButton_save.setFont(self.font)
        self.pushButton_save.setAutoExclusive(False)
        self.pushButton_save.clicked.connect(self.save)

        self.pushButton_segmentation = QtWidgets.QPushButton(self.widget_labels)
        self.pushButton_segmentation.setFont(self.font)
        self.pushButton_segmentation.setAutoExclusive(False)
        self.pushButton_segmentation.clicked.connect(self.segmentation)

        self.putthon_load_tooth_seg_result = QtWidgets.QPushButton(self.widget_labels)
        self.putthon_load_tooth_seg_result.setFont(self.font)
        self.putthon_load_tooth_seg_result.setAutoExclusive(False)
        self.putthon_load_tooth_seg_result.clicked.connect(self.load_tooth_segmentation_result)

        self.putthon_load_alveolar_seg_result = QtWidgets.QPushButton(self.widget_labels)
        self.putthon_load_alveolar_seg_result.setFont(self.font)
        self.putthon_load_alveolar_seg_result.setAutoExclusive(False)
        self.putthon_load_alveolar_seg_result.clicked.connect(self.load_alveolar_segmentation_result)

        self.tool_bar_layout.addWidget(self.widget_labels)
        self.tool_bar_layout.addWidget(self.pushButton_load, Qt.AlignBottom)
        self.tool_bar_layout.addWidget(self.pushButton_save, Qt.AlignBottom)
        self.tool_bar_layout.addWidget(self.pushButton_segmentation, Qt.AlignBottom)
        self.tool_bar_layout.addWidget(self.putthon_load_tooth_seg_result, Qt.AlignBottom)
        self.tool_bar_layout.addWidget(self.putthon_load_alveolar_seg_result, Qt.AlignBottom)

        self.system_layout.addLayout(self.tool_bar_layout, 2)

        MainWindow.setCentralWidget(self.centralwidget)
        menubar_style = """
                   QMenuBar{
                       background-color: rgba(255, 255, 255);
                       border: 1px solid rgba(240, 240, 240);
                   }
                   QMenuBar::item {
                       color: rgb(0, 0, 0);
                       background: rgba(255, 255, 255);
                   }
                   QMenuBar::item:selected {
                       background: rgba(48, 140, 198);
                       color: rgb(255, 255, 255);
                   }
                   QMenuBar::item:pressed {
                       background: rgba(48, 140, 198,0.4);
                   }
               """
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1103, 30))
        self.menubar.setFont(self.font)
        self.menubar.setStyleSheet(menubar_style)

        menu_style = """
                   QMenu {
                       background-color: rgba(255, 255, 255);
                       border: 1px solid rgba(244, 244, 244);
                   }
                   QMenu::item {
                       color: rgb(0, 0, 0);
                       background: rgba(255, 255, 255);
                   }
                   QMenu::item:selected {
                       background: rgba(48, 140, 198);
                       color: rgb(255, 255, 255);
                   }
                   QMenu::item:pressed {
                       background: rgba(48, 140, 198,0.4);
                   }
               """
        self.fileMenu = QtWidgets.QMenu(self.menubar)
        self.fileMenu.setStyleSheet(menu_style)
        MainWindow.setMenuBar(self.menubar)

        self.actionAdd_DiICOM_Data = QtWidgets.QAction(MainWindow)
        self.actionAdd_DiICOM_Data.setObjectName("actionAdd_DiICOM_Data")
        self.actionAdd_DiICOM_Data.triggered.connect(self.add_DICOM_Data)
        self.fileMenu.addAction(self.actionAdd_DiICOM_Data)
        self.menubar.addAction(self.fileMenu.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.fileMenu.setTitle(_translate("MainWindow", "File"))
        self.actionAdd_DiICOM_Data.setText(_translate("MainWindow", "Add DICOM Data"))
        self.label_XY.setText(_translate("MainWindow", "Slice"))
        self.title.setText(_translate("MainWindow", "Contrast Adjustment"))
        self.widget_title.setText(_translate("MainWindow", "Labels"))
        self.labels_title.setText(_translate("MainWindow", "FDI:  "))
        self.window_level.setText(_translate("MainWindow", "Window Level"))
        self.window_width.setText(_translate("MainWindow", "Window Width"))
        self.pushButton_paint.setText(_translate("MainWindow", "Paint"))
        self.pushButton_clear.setText(_translate("MainWindow", "Clear"))
        self.pushButton_undo.setText(_translate("MainWindow", "Undo"))
        self.pushButton_redo.setText(_translate("MainWindow", "Redo"))
        self.pushButton_load.setText(_translate("MainWindow", "Load"))
        self.pushButton_save.setText(_translate("MainWindow", "Save"))
        self.pushButton_segmentation.setText(_translate("MainWindow", "Segmentation"))
        self.putthon_load_tooth_seg_result.setText(_translate("MainWindow", "Load Tooth Segmentation Result"))
        self.putthon_load_alveolar_seg_result.setText(_translate("MainWindow", "Load Alveolar Segmentation Result"))

    def add_DICOM_Data(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(None, "Select Folder")
        if path == "":
            return
        files = os.listdir(path)
        dcm_files_exist = any(file.endswith(".dcm") for file in files)
        if not dcm_files_exist:
            return
        self.clear()
        self.pushButton_paint.setChecked(False)
        self.paint_enable = False
        self.landmarkpath = None
        self.gl_fileIsEmpty = False
        self.pathDicomDir = path
        self.datapath = path
        patientum = self.datapath.split('/')[-1]
        setSubjectName(patientum)
        self.reader = vtk.vtkDICOMImageReader()
        self.reader.SetDirectoryName(self.pathDicomDir)
        self.reader.Update()
        self.dims = self.reader.GetOutput().GetDimensions()
        self.dicomdata, self.header = load(self.pathDicomDir)
        self.value_XY = int(self.dicomdata.shape[2] / 2)
        # -------------------Update------------------------------------------
        self.viewer_XY = vtk.vtkResliceImageViewer()
        self.viewer_XY.SetInputData(self.reader.GetOutput())
        self.viewer_XY.SetupInteractor(self.vtkWidget)
        self.viewer_XY.SetRenderWindow(self.vtkWidget.GetRenderWindow())
        self.viewer_XY.SetSliceOrientationToXY()
        self.viewer_XY.SetSlice(self.value_XY)
        self.camera = self.viewer_XY.GetRenderer().GetActiveCamera()
        self.camera.ParallelProjectionOn()
        self.camera.Zoom(2.5)
        self.viewer_XY.SliceScrollOnMouseWheelOff()
        self.viewer_XY.Render()
        # --------------------------------------------------------------------------------------
        self.wheelforward = MouseWheelForward(self.viewer_XY, self.label_XY, self.verticalSlider_XY)
        self.wheelbackward = MouseWheelBackWard(self.viewer_XY, self.label_XY, self.verticalSlider_XY)
        self.viewer_XY_InteractorStyle = self.viewer_XY.GetInteractorStyle()
        self.viewer_XY_InteractorStyle.AddObserver("MouseWheelForwardEvent", self.wheelforward)
        self.viewer_XY_InteractorStyle.AddObserver("MouseWheelBackwardEvent", self.wheelbackward)

        self.maxSlice = self.viewer_XY.GetSliceMax()
        self.verticalSlider_XY.setMaximum(self.maxSlice)
        self.verticalSlider_XY.setMinimum(0)
        self.verticalSlider_XY.setSingleStep(1)
        self.verticalSlider_XY.setValue(self.value_XY)
        self.verticalSlider_XY.valueChanged.connect(self.verticalSlider_valuechange)
        self.label_XY.setText("Slice %d/%d" % (self.verticalSlider_XY.value(), self.viewer_XY.GetSliceMax()))

        window, level = LevelAndWidth(self)
        self.window_width_slider.setValue(int(window))
        self.window_level_slider.setValue(int(level))

    def paint(self):
        if self.gl_fileIsEmpty == True:
            return
        if self.paint_enable == False:
            print("Start Annotation")
            self.picker = vtk.vtkPointPicker()
            self.picker.PickFromListOn()
            self.left_press = LeftButtonPressEvent(self.picker, self.viewer_XY, self.color_combobox)
            self.viewer_XY_InteractorStyle.AddObserver("LeftButtonPressEvent", self.left_press)
            self.viewer_XY.UpdateDisplayExtent()
            self.viewer_XY.Render()
            self.paint_enable = True
        else:
            print("End Annotation")
            self.viewer_XY_InteractorStyle.RemoveObservers("LeftButtonPressEvent")
            # self.viewer_XY_InteractorStyle.AddObserver("MouseWheelForwardEvent", self.wheelforward)
            # self.viewer_XY_InteractorStyle.AddObserver("MouseWheelBackwardEvent", self.wheelbackward)
            self.paint_enable = False

    def clear(self):
        if self.gl_fileIsEmpty == True:
            return
        try:
            for i in getPaintActors():
                self.viewer_XY.GetRenderer().RemoveActor(i)
            self.viewer_XY.Render()
        except:
            print('Close viewer_XY actor_paint Failed!!!')
        if len(getUndoStack()) > 0:
            current_list = list(set(np.array(getUndoStack())[:, 0]))
            for index in getColorIndexList():
                if index in current_list:
                    self.color_combobox.setItemText(self.color_combobox.findText(str(index) + " √"), str(index))
                self.color_combobox.update()

        clearColorIndexList()
        clearPaintActors()
        clearUndoStack()

    def undo(self):
        if self.gl_fileIsEmpty == True:
            return
        try:
            for i in getPaintActors():
                self.viewer_XY.GetRenderer().RemoveActor(i)
        except:
            print('Close viewer_XY actor_paint Failed!!!')
        if len(getUndoStack()) > 0:
            setRedoStack(getUndoStack().pop())
        current_list = list(set(np.array(getUndoStack())[:, 0])) if len(getUndoStack()) > 0 else []
        for index in getColorIndexList():
            if index not in current_list:
                self.color_combobox.setItemText(self.color_combobox.findText(str(index) + " √"), str(index))
                self.color_combobox.update()
                getColorIndexList().remove(index)
        for point in getUndoStack():
            if point[1] == self.verticalSlider_XY.value():
                print("undo")
                self.paintPoints(point)
        self.viewer_XY.UpdateDisplayExtent()
        self.viewer_XY.Render()

    def redo(self):
        if self.gl_fileIsEmpty == True:
            return
        if len(getRedoStack()) > 0:
            setUndoStack(getRedoStack().pop())
        current_list = list(set(np.array(getUndoStack())[:, 0])) if len(getUndoStack()) > 0 else []
        for index in current_list:
            if index not in getColorIndexList():
                setColorIndexList(index)
                self.color_combobox.setItemText(self.color_combobox.findText(str(index)), str(index) + " √")
            self.color_combobox.update()
        for point in getUndoStack():
            if point[1] == self.verticalSlider_XY.value():
                print("redo")
                self.paintPoints(point)
        self.viewer_XY.UpdateDisplayExtent()
        self.viewer_XY.Render()

    def load(self):
        if self.gl_fileIsEmpty == True:
            return
        dicom_shape = self.viewer_XY.GetInput().GetDimensions()
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Open File", "", "Text Files (*.txt)")
        print(file_path)
        if file_path:
            if len(getUndoStack()) > 0:
                current_list = list(set(np.array(getUndoStack())[:, 0]))
                for index in getColorIndexList():
                    if index in current_list:
                        self.color_combobox.setItemText(self.color_combobox.findText(str(index) + " √"), str(index))
                    self.color_combobox.update()
                    getColorIndexList().remove(index)
            clearUndoStack()
            clearColorIndexList()
            try:
                for i in getPaintActors():
                    self.viewer_XY.GetRenderer().RemoveActor(i)
            except:
                print('Close viewer_XY actor_paint Failed!!!')
            with open(file_path, "r") as file:
                lines = file.readlines()
                for line in lines:
                    point = [int(p) for p in line.strip().split(",")]
                    point[1] = dicom_shape[2] - point[1] - 1
                    point[3] = dicom_shape[1] - point[3] - 1
                    setUndoStack(point)
            info = QtWidgets.QMessageBox()
            info.setIcon(QtWidgets.QMessageBox.Information)
            info.setWindowTitle('Information')
            info.setText('File loaded successfully!')
            info.show()
            info.exec_()
            print(getUndoStack())
            for point in getUndoStack():
                setColorIndexList(point[0])
                if point[1] == self.verticalSlider_XY.value():
                    print("redo")
                    self.paintPoints(point)
            self.viewer_XY.UpdateDisplayExtent()
            self.viewer_XY.Render()
            self.landmarkpath = file_path
            for index in getColorIndexList():
                self.color_combobox.setItemText(self.color_combobox.findText(str(index)), str(index) + " √")
                self.color_combobox.update()

    def save(self):
        if self.gl_fileIsEmpty == True:
            print("file not import")
            return
        dicom_shape = self.viewer_XY.GetInput().GetDimensions()
        if getUndoStack() == []:
            return
        save_file_path = QFileDialog.getSaveFileName(None, "Save File", "", "Text Files (*.txt)")[0]
        if save_file_path:
            with open(save_file_path, "w") as file:
                for point in getUndoStack():
                    point[1] = dicom_shape[2] - point[1] - 1
                    point[3] = dicom_shape[1] - point[3] - 1
                    file.write(",".join(str(p) for p in point) + "\n")
            info = QtWidgets.QMessageBox()
            info.setIcon(QtWidgets.QMessageBox.Information)
            info.setWindowTitle('Information')
            info.setText('File saved successfully!')
            info.show()
            info.exec_()
        self.landmarkpath = save_file_path

    def segmentation(self):
        try:
            if self.gl_fileIsEmpty == True:
                print("file not import")
                return
            if not os.path.exists(self.output_file_path):
                os.makedirs(self.output_file_path)
            patientum = self.datapath.split('/')[-1]
            print("#===============Processing PatientID: " + patientum + "===============#")
            input_cbct_file_path = self.datapath
            input_landmark_textfile_path = self.landmarkpath
            if input_landmark_textfile_path == None:
                print("please load landmark textfile")
                info = QtWidgets.QMessageBox()
                info.setIcon(QtWidgets.QMessageBox.Information)
                info.setWindowTitle('Error')
                info.setText('Please load tooth landmark textfile!')
                info.show()
                info.exec_()
            else:
                info = QtWidgets.QMessageBox()
                info.setIcon(QtWidgets.QMessageBox.Information)
                info.setWindowTitle('Teeth Segmentation in Progress!')
                info.setText('Teeth Segmentation Completed!')
                info.show()
                print("input_cbct_file_path:", input_cbct_file_path)
                print("input_landmark_textfile_path,", input_landmark_textfile_path)
                Step1_CBCT_ToothAlveolar_Seg(input_cbct_file_path, input_landmark_textfile_path, getSubjectName(),
                                             self.output_file_path, self.flag_seg, self.threshold_th,
                                             self.threshold_al, self.smoothfactor, self.erosion_radius_up,
                                             self.erosion_radius_low)
                info.exec_()
        except:
            print("tooth landmark not import!")

    def load_tooth_segmentation_result(self):
        if self.gl_fileIsEmpty == True:
            print("file not import")
            return
        print("load tooth segmentation result")
        type = "Tooth"
        self.tooth_window = UI_ReadSegmentationResult(type=type)
        self.tooth_window.show()
        self.tooth_window.LoadSTL(type=type)

    def load_alveolar_segmentation_result(self):
        if self.gl_fileIsEmpty == True:
            print("file not import")
            return
        print("load alveolar segmentation result")
        type = "Alveolar"
        self.alveolar_window = UI_ReadSegmentationResult(type=type)
        self.alveolar_window.show()
        self.alveolar_window.LoadSTL(type=type)

    def verticalSlider_valuechange(self):
        if self.gl_fileIsEmpty == True:
            return
        else:
            try:
                for i in getPaintActors():
                    self.viewer_XY.GetRenderer().RemoveActor(i)
            except:
                print('Close viewer_XY actor_paint Failed!!!')
            value = self.verticalSlider_XY.value()
            self.viewer_XY.SetSlice(value)
            if getUndoStack() != []:
                for point in getUndoStack():
                    if point[1] == value:
                        print("valuechange")
                        self.paintPoints(point)
            self.viewer_XY.UpdateDisplayExtent()
            self.viewer_XY.Render()
            self.label_XY.setText("Slice %d/%d" % (self.viewer_XY.GetSlice(), self.viewer_XY.GetSliceMax()))

    def window_level_slider_valuechange(self):
        if self.gl_fileIsEmpty == True:
            return
        else:
            self.viewer_XY.SetColorLevel(self.window_level_slider.value())
            self.viewer_XY.SetColorWindow(self.window_width_slider.value())
            self.viewer_XY.Render()

    def window_width_slider_valuechange(self):
        if self.gl_fileIsEmpty == True:
            return
        else:
            self.viewer_XY.SetColorLevel(self.window_level_slider.value())
            self.viewer_XY.SetColorWindow(self.window_width_slider.value())
            self.viewer_XY.Render()

    def update_current_color(self, index):
        self.current_color_data = color_data[index]
        if self.current_color_data[0] == "":
            self.current_color_data = color_data[index + 1]
            self.color_combobox.setCurrentIndex(index + 1)
        setColor(self.current_color_data[0])
        setColorId(self.current_color_data[1])
        setColorIndex(self.color_combobox.currentIndex())
        # self.color_combobox.currentIndex()
        print(self.color_combobox.currentText(), getColor())
        print(self.color_combobox.findText(str(17)))

    def paintPoints(self, point):
        origin = self.viewer_XY.GetInput().GetOrigin()
        spacing = self.viewer_XY.GetInput().GetSpacing()

        print(point)
        label_id = point[0]
        label_color = None
        for data in color_data:
            if data[0] == "":
                continue
            if int(data[1]) == label_id:
                label_color = data[0]
        point_z = point[1]
        point_x = point[2] * spacing[0] + origin[0]
        point_y = point[3] * spacing[1] + origin[1]
        print(label_color, label_id, point_x, point_y, point_z)
        square = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        points.InsertNextPoint(point_x - 0.5, point_y + 0.5, point_z)
        points.InsertNextPoint(point_x + 0.5, point_y + 0.5, point_z)
        points.InsertNextPoint(point_x + 0.5, point_y - 0.5, point_z)
        points.InsertNextPoint(point_x - 0.5, point_y - 0.5, point_z)

        polygon = vtk.vtkPolygon()
        polygon.GetPointIds().SetNumberOfIds(4)
        polygon.GetPointIds().SetId(0, 0)
        polygon.GetPointIds().SetId(1, 1)
        polygon.GetPointIds().SetId(2, 2)
        polygon.GetPointIds().SetId(3, 3)

        cells = vtk.vtkCellArray()
        cells.InsertNextCell(polygon)

        square.SetPoints(points)
        square.SetPolys(cells)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(square)

        rgb = hex_to_rgb(label_color)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(rgb[0], rgb[1], rgb[2])

        setPaintActor(actor)
        self.viewer_XY.GetRenderer().AddActor(actor)


class UI_ReadSegmentationResult(QWidget):
    def __init__(self, type):
        super().__init__()
        self.stl_path = []
        self.resize(1128, 698)
        self.system_layout = QtWidgets.QHBoxLayout(self)
        self.system_layout.setSpacing(6)
        self.reader = vtk.vtkDICOMImageReader()
        self.reader.SetDirectoryName(global_dicomDir_path)
        self.reader.Update()
        self.frame = QtWidgets.QFrame()
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.system_layout.addWidget(self.vtkWidget)
        self.reader_stl_renderer = vtk.vtkRenderer()
        self.reader_stl_renderer.SetBackground(1, 1, 1)
        self.reader_stl_renderer.ResetCamera()
        self.reader_stl_iren = self.vtkWidget.GetRenderWindow().GetInteractor()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.reader_stl_renderer)
        self.reader_stl_style = vtk.vtkInteractorStyleTrackballCamera()
        self.reader_stl_style.SetDefaultRenderer(self.reader_stl_renderer)
        self.reader_stl_style.EnabledOn()
        self.vtkWidget.Render()

        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("Widget", type + " Segmenation Result"))

    def LoadSTL(self, type):
        print("subject_name: " + getSubjectName())
        datanames = os.listdir("./output")
        self.stl_path = []
        if type == "Tooth":
            for dataname in datanames:
                if dataname.endswith('.stl') and ("LowerTooth" in dataname or "UpperTooth" in dataname):
                    self.stl_path.append(dataname)
        elif type == "Alveolar":
            for dataname in datanames:
                if dataname.endswith('.stl') and ("LowerAll" in dataname or "UpperAll" in dataname):
                    self.stl_path.append(dataname)
        print(self.stl_path)

        for filename in self.stl_path:
            filename = './output/' + filename
            bounds = self.reader.GetOutput().GetBounds()
            self.center0 = (bounds[1] + bounds[0]) / 2.0
            self.center1 = (bounds[3] + bounds[2]) / 2.0
            self.center2 = (bounds[5] + bounds[4]) / 2.0
            transform = vtk.vtkTransform()
            transform.Translate(self.center0, self.center1, self.center2)
            reader = vtk.vtkSTLReader()
            reader.SetFileName(filename)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(reader.GetOutputPort())
            actor = vtk.vtkLODActor()
            actor.SetMapper(mapper)
            actor.SetUserTransform(transform)
            actor.GetProperty().SetColor(0.9, 0.8, 0.2)
            self.reader_stl_iren.SetInteractorStyle(self.reader_stl_style)
            self.reader_stl_renderer.AddActor(actor)
        self.vtkWidget.Render()
        self.reader_stl_renderer.ResetCamera()
        self.reader_stl_iren.Initialize()


class MouseWheelForward():
    def __init__(self, viewer, labeltext, verticalslider):
        self.view = viewer
        self.actor = self.view.GetImageActor()
        self.iren = self.view.GetRenderWindow().GetInteractor()
        self.render = self.view.GetRenderer()
        self.image = self.view.GetInput()
        self.origin = self.image.GetOrigin()
        self.spacing = self.image.GetSpacing()
        self.label = labeltext
        self.verticalslider = verticalslider

    def __call__(self, caller, ev):
        self.sliceXY = self.view.GetSlice()
        self.verticalslider.setValue(self.sliceXY)
        self.sliceXY += 1
        if self.sliceXY <= self.view.GetSliceMax():
            self.verticalslider.setValue(self.sliceXY)
            self.label.setText("Slice %d/%d" % (self.view.GetSlice(), self.view.GetSliceMax()))
        else:
            self.sliceXY = self.view.GetSliceMax()
            self.verticalslider.setValue(self.view.GetSliceMax())
            self.label.setText("Slice %d/%d" % (self.view.GetSlice(), self.view.GetSliceMax()))


class MouseWheelBackWard():
    def __init__(self, viewer, labeltext, verticalslider):
        self.start = []
        self.view = viewer
        self.actor = self.view.GetImageActor()
        self.iren = self.view.GetRenderWindow().GetInteractor()
        self.render = self.view.GetRenderer()
        self.image = self.view.GetInput()
        self.origin = self.image.GetOrigin()
        self.spacing = self.image.GetSpacing()
        self.label = labeltext
        self.verticalslider = verticalslider

    def __call__(self, caller, ev):
        self.sliceXY = self.view.GetSlice()
        self.verticalslider.setValue(self.sliceXY)
        self.sliceXY -= 1
        if self.sliceXY >= 0:
            self.verticalslider.setValue(self.sliceXY)
            self.label.setText("Slice %d/%d" % (self.view.GetSlice(), self.view.GetSliceMax()))
        else:
            self.sliceXY = 0
            self.verticalslider.setValue(0)
            self.label.setText("Slice %d/%d" % (self.view.GetSlice(), self.view.GetSliceMax()))


class LeftButtonPressEvent():
    def __init__(self, picker, viewer, color_combobox):
        self.picker = picker
        self.start = []
        self.view = viewer
        self.color_combobox = color_combobox
        self.actor = self.view.GetImageActor()
        self.iren = self.view.GetRenderWindow().GetInteractor()
        self.render = self.view.GetRenderer()
        self.image = self.view.GetInput()
        self.origin = self.image.GetOrigin()
        self.spacing = self.image.GetSpacing()
        self.square_actor = None
        self.imageshape = self.view.GetInput().GetDimensions()

    def __call__(self, caller, ev):
        self.picker.AddPickList(self.actor)
        self.picker.SetTolerance(0.01)
        self.picker.Pick(self.iren.GetEventPosition()[0], self.iren.GetEventPosition()[1], 0, self.render)
        self.start = self.picker.GetPickPosition()
        print(self.start)

        point_x = int((self.start[0] - self.origin[0]) / self.spacing[0])
        point_y = int((self.start[1] - self.origin[1]) / self.spacing[1])
        point_z = int((self.start[2] - self.origin[2]) / self.spacing[2])
        if point_x < 0 or point_x > self.imageshape[0] or point_y < 0 or point_y > self.imageshape[1]:
            return

        self.start_pos = [point_x, point_y, point_z]
        print(self.start_pos)

        square = vtk.vtkPolyData()

        points = vtk.vtkPoints()
        points.InsertNextPoint(self.start[0] - 0.5, self.start[1] + 0.5, point_z)
        points.InsertNextPoint(self.start[0] + 0.5, self.start[1] + 0.5, point_z)
        points.InsertNextPoint(self.start[0] + 0.5, self.start[1] - 0.5, point_z)
        points.InsertNextPoint(self.start[0] - 0.5, self.start[1] - 0.5, point_z)

        polygon = vtk.vtkPolygon()
        polygon.GetPointIds().SetNumberOfIds(4)
        polygon.GetPointIds().SetId(0, 0)
        polygon.GetPointIds().SetId(1, 1)
        polygon.GetPointIds().SetId(2, 2)
        polygon.GetPointIds().SetId(3, 3)

        cells = vtk.vtkCellArray()
        cells.InsertNextCell(polygon)

        square.SetPoints(points)
        square.SetPolys(cells)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(square)

        rgb = hex_to_rgb(getColor())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(rgb[0], rgb[1], rgb[2])

        setUndoStack([int(getColorId()), point_z, point_x, point_y])
        self.square_actor = actor
        setPaintActor(self.square_actor)
        setColorIndexList(int(getColorId()))
        self.color_combobox.setItemText(getColorIndex(), getColorId() + " √")
        self.ren = self.view.GetRenderer()
        self.ren.AddActor(actor)
        self.iren.Render()
        self.render.Render()
        self.view.UpdateDisplayExtent()
        self.view.Render()


if __name__ == "__main__":
    import sys

    global_dicomDir_path = os.path.abspath(os.path.dirname(__file__))
    app = QtWidgets.QApplication(sys.argv)
    Widget = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(Widget)
    Widget.show()
    sys.exit(app.exec_())
