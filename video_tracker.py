import sys
import os
import cv2
import numpy as np
import pandas as pd
import shutil
import json
import math
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QStatusBar, QMessageBox, QFrame, QSizePolicy,
                             QInputDialog)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QPoint, QRect
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush

# -----------------------------------------------------------------------------
# Dependency Installation Instructions:
# pip install PyQt5 opencv-python numpy pandas
# -----------------------------------------------------------------------------

class VideoManager:
    """
    Handles video loading and frame extraction using OpenCV.
    """
    def __init__(self):
        self.cap = None
        self.total_frames = 0
        self.fps = 0.0
        self.width = 0
        self.height = 0
        self.current_frame_idx = 0
        self.video_path = ""

    def load_video(self, path):
        """Loads a video file and extracts metadata."""
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            return False
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_frame_idx = 0
        self.video_path = path
        return True

    def get_frame(self, frame_idx):
        """Retrieves a specific frame by index."""
        if self.cap is None:
            return None
        
        # Optimize: Only set pos if not sequential (though safe to always set)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret:
            # Convert BGR to RGB for Qt
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
        return None

    def release(self):
        if self.cap:
            self.cap.release()

class AnnotationCanvas(QLabel):
    """
    Widget to display video frames and handle mouse interactions for annotations.
    """
    # Signals to communicate with MainWindow
    annotation_created = pyqtSignal(dict) # {mode, x, y, w, h}
    calibration_created = pyqtSignal(int, int, int, int) # x1, y1, x2, y2 in video coords

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #222;") # Dark background for canvas

        self.current_frame_image = None # QPixmap
        self.mode = "point" # "point", "bbox", or "calibrate"
        
        # Drawing state
        self.drawing = False
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.current_annotation = None # Existing annotation for this frame (dict)
        
        # Calibration line to show (video coords: x1, y1, x2, y2) or None
        self.calibration_line = None
        
        # Scaling factors (displayed size vs actual video size)
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0

    def set_frame(self, frame_rgb, annotation=None, calibration_line=None):
        """Displays a new frame and any existing annotation."""
        self.current_annotation = annotation
        self.calibration_line = calibration_line
        
        if frame_rgb is not None:
            height, width, channel = frame_rgb.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.current_frame_image = QPixmap.fromImage(q_img)
        else:
            self.current_frame_image = None
            
        self.update() # Trigger paintEvent

    def paintEvent(self, event):
        """Draws the video frame, existing annotations, and temporary drawing shapes."""
        super().paintEvent(event)
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if self.current_frame_image:
            # Letterbox scaling (keep aspect ratio)
            w = self.width()
            h = self.height()
            
            img_w = self.current_frame_image.width()
            img_h = self.current_frame_image.height()
            
            # Calculate scaled dimensions
            scale_w = w / img_w
            scale_h = h / img_h
            self.scale_factor = min(scale_w, scale_h)
            
            scaled_w = int(img_w * self.scale_factor)
            scaled_h = int(img_h * self.scale_factor)
            
            # Center the image
            self.offset_x = (w - scaled_w) // 2
            self.offset_y = (h - scaled_h) // 2
            
            target_rect = QRect(self.offset_x, self.offset_y, scaled_w, scaled_h)
            painter.drawPixmap(target_rect, self.current_frame_image)

            # Draw saved annotation for this frame
            if self.current_annotation:
                self._draw_annotation(painter, self.current_annotation, is_preview=False)

            # Draw temporary annotation (while dragging bbox)
            if self.drawing and self.mode == "bbox":
                rect = self._get_rect_from_points(self.start_point, self.end_point)
                pen = QPen(Qt.green, 2, Qt.SolidLine)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(rect)

            # Draw temporary calibration line (while dragging)
            if self.drawing and self.mode == "calibrate":
                pen = QPen(Qt.cyan, 2, Qt.DashLine)
                painter.setPen(pen)
                painter.drawLine(self.start_point, self.end_point)

            # Draw saved calibration line
            if self.calibration_line:
                cx1, cy1, cx2, cy2 = self.calibration_line
                sx1 = int(cx1 * self.scale_factor) + self.offset_x
                sy1 = int(cy1 * self.scale_factor) + self.offset_y
                sx2 = int(cx2 * self.scale_factor) + self.offset_x
                sy2 = int(cy2 * self.scale_factor) + self.offset_y
                pen = QPen(Qt.cyan, 2, Qt.SolidLine)
                painter.setPen(pen)
                painter.drawLine(sx1, sy1, sx2, sy2)
                # Small circles at endpoints
                painter.setBrush(QColor(0, 255, 255, 150))
                painter.drawEllipse(QPoint(sx1, sy1), 4, 4)
                painter.drawEllipse(QPoint(sx2, sy2), 4, 4)

    def _draw_annotation(self, painter, ann, is_preview=False):
        """Helper to draw a specific annotation dict."""
        # Convert video coordinates back to screen coordinates
        x = int(ann['x'] * self.scale_factor) + self.offset_x
        y = int(ann['y'] * self.scale_factor) + self.offset_y
        
        if ann['mode'] == 'point':
            pen_color = Qt.red
            brush_color = QColor(255, 0, 0, 150)
            radius = 5
            
            painter.setPen(QPen(pen_color, 2))
            painter.setBrush(brush_color)
            painter.drawEllipse(QPoint(x, y), radius, radius)
            
        elif ann['mode'] == 'bbox':
            w = int(ann['w'] * self.scale_factor)
            h = int(ann['h'] * self.scale_factor)
            
            pen_color = Qt.green
            painter.setPen(QPen(pen_color, 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(x, y, w, h)

    def mousePressEvent(self, event):
        if not self.current_frame_image:
            return
            
        if event.button() == Qt.LeftButton:
            # Check if click is within the video image
            pos = event.pos()
            if self._is_within_image(pos):
                self.drawing = True
                self.start_point = pos
                self.end_point = pos
                
                if self.mode == "point":
                    self._commit_annotation(pos)
                    self.drawing = False # No drag for point
                # calibrate and bbox both use drag, handled on release

    def mouseMoveEvent(self, event):
        if self.drawing and self.mode in ("bbox", "calibrate"):
            self.end_point = event.pos()
            self.update() # Repaint for preview

    def mouseReleaseEvent(self, event):
        if not self.drawing or event.button() != Qt.LeftButton:
            return

        self.drawing = False
        self.end_point = event.pos()

        if self.mode == "bbox":
            # Normalize rect
            rect = self._get_rect_from_points(self.start_point, self.end_point)
            
            # If rect is too small, ignore
            if rect.width() > 5 and rect.height() > 5:
                video_rect = self._screen_to_video_rect(rect)
                if video_rect:
                    self.annotation_created.emit({
                        "mode": "bbox",
                        "x": video_rect[0],
                        "y": video_rect[1],
                        "w": video_rect[2],
                        "h": video_rect[3]
                    })

        elif self.mode == "calibrate":
            # Convert both endpoints to video coordinates
            vx1, vy1 = self._screen_to_video_coords(self.start_point)
            vx2, vy2 = self._screen_to_video_coords(self.end_point)
            pixel_dist = math.hypot(vx2 - vx1, vy2 - vy1)
            if pixel_dist > 5:  # Ignore tiny lines
                self.calibration_created.emit(vx1, vy1, vx2, vy2)

    def _commit_annotation(self, pos):
        """Converts screen point to video coordinates and emits signal."""
        vx, vy = self._screen_to_video_coords(pos)
        self.annotation_created.emit({
            "mode": "point",
            "x": vx,
            "y": vy,
            "w": 0,
            "h": 0
        })
        self.update()

    def _is_within_image(self, pos):
        # Allow clicking a bit outside for usability? Strict for now.
        # But we must only map coords if inside or valid.
        # Actually logic is simpler if we clamp subsequent moves.
        # Check simple bounds of the drawn pixmap
        img_rect = QRect(self.offset_x, self.offset_y, 
                         int(self.current_frame_image.width() * self.scale_factor),
                         int(self.current_frame_image.height() * self.scale_factor))
        return img_rect.contains(pos)

    def _screen_to_video_coords(self, pos):
        """Converts screen x,y to video pixel coordinates."""
        rel_x = pos.x() - self.offset_x
        rel_y = pos.y() - self.offset_y
        
        vid_x = int(rel_x / self.scale_factor)
        vid_y = int(rel_y / self.scale_factor)
        
        # Clamp
        vid_x = max(0, min(vid_x, self.current_frame_image.width() - 1))
        vid_y = max(0, min(vid_y, self.current_frame_image.height() - 1))
        
        return vid_x, vid_y

    def _screen_to_video_rect(self, screen_rect):
        """Converts screen QRect to (x, y, w, h) in video coordinates."""
        top_left = screen_rect.topLeft()
        bottom_right = screen_rect.bottomRight()
        
        x1, y1 = self._screen_to_video_coords(top_left)
        x2, y2 = self._screen_to_video_coords(bottom_right)
        
        return (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))

    def _get_rect_from_points(self, p1, p2):
        return QRect(p1, p2).normalized()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt5 Video Tracker")
        self.resize(1200, 800)

        self.video_manager = VideoManager()
        self.annotations = {} # { frame_idx: {mode, x, y, w, h} }
        self.current_frame = 0

        # Calibration state
        # calibration_line: (x1, y1, x2, y2) in video pixel coords
        self.calibration_line = None
        # pixels_per_meter: computed from calibration
        self.pixels_per_meter = None
        # The mode we return to after calibration
        self._pre_calibrate_mode = "point"

        self.setup_ui()
        self.update_ui_state(has_video=False)

    def setup_ui(self):
        # Central Widget & Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # ---------------------------------------------------------------------
        # Toolbar
        # ---------------------------------------------------------------------
        toolbar_layout = QHBoxLayout()
        
        self.btn_open = QPushButton("Open Video")
        self.btn_open.setFocusPolicy(Qt.NoFocus)
        self.btn_open.clicked.connect(self.open_video)
        
        self.btn_save_csv = QPushButton("Save CSV")
        self.btn_save_csv.setFocusPolicy(Qt.NoFocus)
        self.btn_save_csv.clicked.connect(self.save_csv)
        
        self.btn_export_video = QPushButton("Export Annotated Video")
        self.btn_export_video.setFocusPolicy(Qt.NoFocus)
        self.btn_export_video.clicked.connect(self.export_video)
        
        # Mode Toggle
        self.lbl_mode = QLabel("Mode:")
        self.btn_mode_toggle = QPushButton("Point")
        self.btn_mode_toggle.setFocusPolicy(Qt.NoFocus)
        self.btn_mode_toggle.setCheckable(True)
        self.btn_mode_toggle.clicked.connect(self.toggle_mode)
        
        self.btn_clear_frame = QPushButton("Clear Current")
        self.btn_clear_frame.setFocusPolicy(Qt.NoFocus)
        self.btn_clear_frame.clicked.connect(self.clear_current_annotation)
        
        self.btn_clear_all = QPushButton("Clear All")
        self.btn_clear_all.setFocusPolicy(Qt.NoFocus)
        self.btn_clear_all.clicked.connect(self.clear_all_annotations)

        # Calibration button
        self.btn_calibrate = QPushButton("Calibrate Scale")
        self.btn_calibrate.setFocusPolicy(Qt.NoFocus)
        self.btn_calibrate.clicked.connect(self.enter_calibrate_mode)

        self.lbl_calibration_info = QLabel("")
        self.lbl_calibration_info.setStyleSheet("color: cyan; font-weight: bold;")

        # Add widgets to toolbar
        toolbar_layout.addWidget(self.btn_open)
        toolbar_layout.addWidget(self.btn_save_csv)
        toolbar_layout.addWidget(self.btn_export_video)
        
        # Project Management
        self.btn_save_project = QPushButton("Save Project Folder")
        self.btn_save_project.setFocusPolicy(Qt.NoFocus)
        self.btn_save_project.clicked.connect(self.save_project_folder)
        
        self.btn_open_project = QPushButton("Open Project Folder")
        self.btn_open_project.setFocusPolicy(Qt.NoFocus)
        self.btn_open_project.clicked.connect(self.open_project_folder)
        
        toolbar_layout.addWidget(self.btn_save_project)
        toolbar_layout.addWidget(self.btn_open_project)

        toolbar_layout.addStretch() # Spacer
        toolbar_layout.addWidget(self.btn_calibrate)
        toolbar_layout.addWidget(self.lbl_calibration_info)
        toolbar_layout.addSpacing(10)
        toolbar_layout.addWidget(self.lbl_mode)
        toolbar_layout.addWidget(self.btn_mode_toggle)
        toolbar_layout.addSpacing(20)
        toolbar_layout.addWidget(self.btn_clear_frame)
        toolbar_layout.addWidget(self.btn_clear_all)

        main_layout.addLayout(toolbar_layout)

        # ---------------------------------------------------------------------
        # Canvas Area
        # ---------------------------------------------------------------------
        # We put the canvas in a frame for better visual separation
        frame_canvas = QFrame()
        frame_canvas.setFrameShape(QFrame.StyledPanel)
        frame_canvas.setStyleSheet("background-color: #333;")
        canvas_layout = QVBoxLayout(frame_canvas)
        canvas_layout.setContentsMargins(0,0,0,0)
        
        self.canvas = AnnotationCanvas()
        self.canvas.annotation_created.connect(self.add_annotation)
        self.canvas.calibration_created.connect(self.finish_calibration)
        canvas_layout.addWidget(self.canvas)
        
        main_layout.addWidget(frame_canvas, stretch=1)

        # ---------------------------------------------------------------------
        # Status Bar
        # ---------------------------------------------------------------------
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label)

    def update_ui_state(self, has_video):
        """Enables/disables buttons based on video state."""
        self.btn_save_csv.setEnabled(has_video)
        self.btn_export_video.setEnabled(has_video)
        self.btn_save_project.setEnabled(has_video)
        self.btn_calibrate.setEnabled(has_video)
        self.btn_mode_toggle.setEnabled(has_video)
        self.btn_clear_frame.setEnabled(has_video)
        self.btn_clear_all.setEnabled(has_video)
        
        if not has_video:
            self.canvas.setText("No Video Loaded. Click 'Open Video' or 'Open Project'.")
            self.canvas.setStyleSheet("QLabel { color : white; font-size: 16px; background-color: #222; }")
        else:
            self.canvas.setText("")
        
        self._update_calibration_label()

    def toggle_mode(self):
        if self.btn_mode_toggle.isChecked():
            self.btn_mode_toggle.setText("Bounding Box")
            self.canvas.mode = "bbox"
        else:
            self.btn_mode_toggle.setText("Point")
            self.canvas.mode = "point"
        self.update_status()

    # -------------------------------------------------------------------------
    # Calibration
    # -------------------------------------------------------------------------
    def enter_calibrate_mode(self):
        """Switch to calibrate mode so the next drag draws a reference line."""
        self._pre_calibrate_mode = self.canvas.mode
        self.canvas.mode = "calibrate"
        self.status_bar.showMessage(
            "CALIBRATE: Draw a line across a known-length object, then enter its real size.",
            10000)

    def finish_calibration(self, x1, y1, x2, y2):
        """Called when user finishes drawing the calibration line."""
        pixel_dist = math.hypot(x2 - x1, y2 - y1)

        real_length, ok = QInputDialog.getDouble(
            self, "Calibration",
            "Enter the real-life length of this line (in meters):",
            value=1.0, min=0.0001, max=100000.0, decimals=4)

        if ok and real_length > 0:
            self.calibration_line = (x1, y1, x2, y2)
            self.pixels_per_meter = pixel_dist / real_length
            self.status_bar.showMessage(
                f"Calibrated: {self.pixels_per_meter:.2f} px/m  "
                f"({pixel_dist:.1f} px = {real_length} m)", 5000)
        else:
            self.status_bar.showMessage("Calibration cancelled.", 3000)

        # Restore previous mode
        self.canvas.mode = self._pre_calibrate_mode
        self.show_frame()
        self._update_calibration_label()

    def _update_calibration_label(self):
        if self.pixels_per_meter:
            self.lbl_calibration_info.setText(
                f"Scale: {self.pixels_per_meter:.1f} px/m")
        else:
            self.lbl_calibration_info.setText("")

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if path:
            if self.video_manager.load_video(path):
                self.current_frame = 0
                self.annotations = {}
                self.calibration_line = None
                self.pixels_per_meter = None
                self.update_ui_state(has_video=True)
                self.show_frame()
                self.update_status()
            else:
                QMessageBox.critical(self, "Error", "Could not load video file.")

    def show_frame(self):
        """Gets frame from manager and updates canvas."""
        frame = self.video_manager.get_frame(self.current_frame)
        if frame is not None:
            ann = self.annotations.get(self.current_frame)
            self.canvas.set_frame(frame, annotation=ann,
                                  calibration_line=self.calibration_line)
        else:
            self.status_bar.showMessage("Error reading frame", 3000)

    def add_annotation(self, data):
        """Callback from canvas when user draws something."""
        self.annotations[self.current_frame] = data
        self.show_frame() # Redraw to show the committed annotation
        # Advance frame automatically? Requirement: "Save current annotation automatically -> Load next frame"? 
        # User requirement says: "When moving frames: Save current annotation...". 
        # But also: "Point mode: User clicks -> save (x,y)".
        # It's usually better flow to stay on frame to confirm visually, user can hit Space to move.
        # But requirement says "User clicks on frame -> save (x,y) for current frame." 
        # I will just save it. Navigation handles the moving.
        self.update_status()

    def clear_current_annotation(self):
        if self.current_frame in self.annotations:
            del self.annotations[self.current_frame]
            self.show_frame()

    def clear_all_annotations(self):
        confirm = QMessageBox.question(self, "Confirm", "Clear ALL annotations?", QMessageBox.Yes | QMessageBox.No)
        if confirm == QMessageBox.Yes:
            self.annotations = {}
            self.calibration_line = None
            self.pixels_per_meter = None
            self._update_calibration_label()
            self.show_frame()

    # -------------------------------------------------------------------------
    # Navigation
    # -------------------------------------------------------------------------
    def keyPressEvent(self, event):
        if not self.video_manager.cap:
            return

        key = event.key()
        
        if key == Qt.Key_Space or key == Qt.Key_Right:
            self.next_frame()
        elif key == Qt.Key_Left:
            self.prev_frame()
        elif key == Qt.Key_Backspace or key == Qt.Key_Delete:
            self.clear_current_annotation()
        else:
            super().keyPressEvent(event)

    def next_frame(self):
        if self.current_frame < self.video_manager.total_frames - 1:
            self.current_frame += 1
            self.show_frame()
            self.update_status()

    def prev_frame(self):
        if self.current_frame > 0:
            self.current_frame -= 1
            self.show_frame()
            self.update_status()

    def update_status(self):
        info = f"Frame: {self.current_frame + 1} / {self.video_manager.total_frames}"
        mode_str = f"Mode: {self.canvas.mode.upper()}"
        ann_count = f"Annotations: {len(self.annotations)}"
        cal_str = "Calibrated" if self.pixels_per_meter else "Not calibrated"
        self.status_label.setText(f"{info} | {mode_str} | {ann_count} | {cal_str}")

    # -------------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------------
    def _build_annotations_dataframe(self):
        """Build a DataFrame from annotations, with optional real-world columns."""
        data = []
        ppm = self.pixels_per_meter
        sorted_frames = sorted(self.annotations.keys())
        prev_x, prev_y = None, None
        cumulative_dist_m = 0.0

        for fid in sorted_frames:
            ann = self.annotations[fid]
            row = {
                "frame": fid,
                "mode": ann["mode"],
                "x": ann["x"],
                "y": ann["y"],
                "width": ann["w"],
                "height": ann["h"],
            }

            if ppm:
                row["x_m"] = round(ann["x"] / ppm, 6)
                row["y_m"] = round(ann["y"] / ppm, 6)
                if ann["mode"] == "bbox":
                    row["width_m"] = round(ann["w"] / ppm, 6)
                    row["height_m"] = round(ann["h"] / ppm, 6)
                else:
                    row["width_m"] = 0
                    row["height_m"] = 0

                # Distance from previous annotated frame (point mode)
                if ann["mode"] == "point" and prev_x is not None:
                    d_px = math.hypot(ann["x"] - prev_x, ann["y"] - prev_y)
                    d_m = d_px / ppm
                    cumulative_dist_m += d_m
                    row["dist_from_prev_m"] = round(d_m, 6)
                else:
                    row["dist_from_prev_m"] = 0
                row["cumulative_dist_m"] = round(cumulative_dist_m, 6)
            else:
                row["x_m"] = ""
                row["y_m"] = ""
                row["width_m"] = ""
                row["height_m"] = ""
                row["dist_from_prev_m"] = ""
                row["cumulative_dist_m"] = ""

            if ann["mode"] == "point":
                prev_x, prev_y = ann["x"], ann["y"]

            data.append(row)

        return pd.DataFrame(data)

    def save_csv(self):
        if not self.annotations:
            QMessageBox.information(self, "Info", "No annotations to save.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv)")
        if path:
            df = self._build_annotations_dataframe()
            try:
                df.to_csv(path, index=False)
                QMessageBox.information(self, "Success", f"Saved {len(df)} annotations to {path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save CSV: {str(e)}")

    def export_video(self):
        if not self.video_manager.cap:
            return

        output_path, _ = QFileDialog.getSaveFileName(self, "Export Video", "annotated_output.mp4", "MP4 Files (*.mp4)")
        if not output_path:
            return

        ppm = self.pixels_per_meter
        fps = self.video_manager.fps

        # Pre-compute per-frame real-world data so we can show cumulative info
        # as each frame is rendered.
        sorted_ann_frames = sorted(self.annotations.keys())
        frame_realdata = {}  # frame_idx -> {pos_m, step_m, cumul_m, speed_mps}
        prev_x, prev_y = None, None
        prev_fid = None
        cumul_m = 0.0
        trail_points = []  # list of (x, y) in pixel coords for path drawing

        for fid in sorted_ann_frames:
            ann = self.annotations[fid]
            rd = {"pos_m": None, "step_m": 0.0, "cumul_m": 0.0, "speed_mps": 0.0}

            if ann["mode"] == "point":
                trail_points_snapshot = list(trail_points) + [(ann["x"], ann["y"])]

                if ppm:
                    rd["pos_m"] = (ann["x"] / ppm, ann["y"] / ppm)

                    if prev_x is not None:
                        d_px = math.hypot(ann["x"] - prev_x, ann["y"] - prev_y)
                        d_m = d_px / ppm
                        cumul_m += d_m
                        rd["step_m"] = d_m

                        # Speed: distance / time between frames
                        if fps > 0 and prev_fid is not None:
                            dt = (fid - prev_fid) / fps
                            if dt > 0:
                                rd["speed_mps"] = d_m / dt

                    rd["cumul_m"] = cumul_m

                prev_x, prev_y = ann["x"], ann["y"]
                prev_fid = fid
                trail_points.append((ann["x"], ann["y"]))
            else:
                trail_points_snapshot = list(trail_points)

            rd["trail"] = trail_points_snapshot
            frame_realdata[fid] = rd

        # For frames between annotations, carry the last known data forward
        # so the HUD doesn't disappear. Build a lookup.
        last_rd = {"pos_m": None, "step_m": 0.0, "cumul_m": 0.0,
                    "speed_mps": 0.0, "trail": []}

        # Prepare writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps,
                              (self.video_manager.width, self.video_manager.height))

        self.status_bar.showMessage("Exporting video... please wait.")
        QApplication.processEvents()

        temp_cap = cv2.VideoCapture(self.video_manager.video_path)
        frame_idx = 0
        total = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vid_w = self.video_manager.width
        vid_h = self.video_manager.height

        while True:
            ret, frame = temp_cap.read()
            if not ret:
                break

            # Update last_rd if this frame has data
            if frame_idx in frame_realdata:
                last_rd = frame_realdata[frame_idx]

            # -- Draw annotation overlays --
            if frame_idx in self.annotations:
                ann = self.annotations[frame_idx]
                if ann['mode'] == 'point':
                    cv2.circle(frame, (ann['x'], ann['y']), 5, (0, 0, 255), -1)
                elif ann['mode'] == 'bbox':
                    x, y, w, h = ann['x'], ann['y'], ann['w'], ann['h']
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # -- Draw trailing path --
            trail = last_rd.get("trail", [])
            if len(trail) > 1:
                for i in range(1, len(trail)):
                    pt1 = (int(trail[i-1][0]), int(trail[i-1][1]))
                    pt2 = (int(trail[i][0]), int(trail[i][1]))
                    cv2.line(frame, pt1, pt2, (255, 100, 100), 2)
                # Draw dots on trail
                for pt in trail:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, (255, 150, 150), -1)

            # -- Draw calibration line --
            if self.calibration_line:
                cx1, cy1, cx2, cy2 = self.calibration_line
                cv2.line(frame, (cx1, cy1), (cx2, cy2), (255, 255, 0), 2)

            # -- Draw HUD panel (top-left) --
            # Semi-transparent background for readability
            panel_h = 160 if ppm else 50
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (380, panel_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            y_text = 25
            cv2.putText(frame, f"Frame: {frame_idx}", (10, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            y_text += 28

            if ppm:
                cv2.putText(frame, f"Scale: {ppm:.1f} px/m",
                            (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 255, 255), 1)
                y_text += 25

                pos = last_rd.get("pos_m")
                if pos:
                    cv2.putText(frame, f"Pos: ({pos[0]:.3f}, {pos[1]:.3f}) m",
                                (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (200, 200, 255), 1)
                    y_text += 25

                step = last_rd.get("step_m", 0)
                cumul = last_rd.get("cumul_m", 0)
                speed = last_rd.get("speed_mps", 0)

                cv2.putText(frame, f"Step: {step:.4f} m | Total: {cumul:.4f} m",
                            (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (100, 255, 100), 1)
                y_text += 25

                cv2.putText(frame, f"Speed: {speed:.4f} m/s",
                            (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (100, 200, 255), 1)

            out.write(frame)
            frame_idx += 1

            if frame_idx % 50 == 0:
                self.status_bar.showMessage(f"Exporting... {frame_idx}/{total}")
                QApplication.processEvents()

        temp_cap.release()
        out.release()
        self.status_bar.showMessage("Export complete!", 5000)
        QMessageBox.information(self, "Success", f"Video exported to {output_path}")

    # -------------------------------------------------------------------------
    # Project Management
    # -------------------------------------------------------------------------
    def save_project_folder(self):
        if not self.video_manager.cap or not self.video_manager.video_path:
            QMessageBox.warning(self, "Warning", "No video loaded to save.")
            return

        # Let user choose a directory. 
        # Ideally, we create a new folder inside it.
        parent_dir = QFileDialog.getExistingDirectory(self, "Select Parent Directory to Create Project Folder")
        if not parent_dir:
            return

        # Ask for project name
        # Simple input dialog would be nice, but standard QFileDialog doesn't do "New Folder Name" well.
        # We'll assume the user created the folder in the dialog, OR we can just save directly into the chosen dir.
        # Let's verify if the chosen dir is empty or ask.
        # Actually simplest: "Save Project" -> choose a folder (e.g. "MyProject"). We dump files in there.
        project_dir = parent_dir # User should have created/selected the specific folder

        try:
            # 1. Save CSV
            csv_path = os.path.join(project_dir, "annotations.csv")
            df = self._build_annotations_dataframe()
            df.to_csv(csv_path, index=False)

            # 2. Save calibration metadata
            meta = {}
            if self.calibration_line:
                meta["calibration_line"] = list(self.calibration_line)
            if self.pixels_per_meter:
                meta["pixels_per_meter"] = self.pixels_per_meter
            meta_path = os.path.join(project_dir, "calibration.json")
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            # 3. Copy Video
            original_filename = os.path.basename(self.video_manager.video_path)
            dest_video_path = os.path.join(project_dir, original_filename)
            
            if os.path.abspath(self.video_manager.video_path) != os.path.abspath(dest_video_path):
                self.status_bar.showMessage("Copying video file... please wait.")
                QApplication.processEvents()
                shutil.copy2(self.video_manager.video_path, dest_video_path)
            
            self.status_bar.showMessage("Project saved successfully!", 5000)
            QMessageBox.information(self, "Success", f"Project saved to:\n{project_dir}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save project: {str(e)}")
            self.status_bar.showMessage("Error saving project.")

    def open_project_folder(self):
        project_dir = QFileDialog.getExistingDirectory(self, "Select Project Folder")
        if not project_dir:
            return

        try:
            # Find video file
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
            video_file = None
            for f in os.listdir(project_dir):
                if any(f.lower().endswith(ext) for ext in video_extensions):
                    video_file = os.path.join(project_dir, f)
                    break
            
            if not video_file:
                QMessageBox.critical(self, "Error", "No video file found in this folder.")
                return

            # Find CSV
            csv_file = os.path.join(project_dir, "annotations.csv")
            if not os.path.exists(csv_file):
                # Try finding any csv?
                csv_candidates = [f for f in os.listdir(project_dir) if f.endswith('.csv')]
                if csv_candidates:
                    csv_file = os.path.join(project_dir, csv_candidates[0])
                else:
                    QMessageBox.warning(self, "Warning", "No annotations.csv found. Opening video only.")
                    csv_file = None

            # Load Video
            if self.video_manager.load_video(video_file):
                self.current_frame = 0
                self.annotations = {}
                self.calibration_line = None
                self.pixels_per_meter = None
                
                # Load calibration metadata if it exists
                meta_path = os.path.join(project_dir, "calibration.json")
                if os.path.exists(meta_path):
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    if "calibration_line" in meta:
                        self.calibration_line = tuple(meta["calibration_line"])
                    if "pixels_per_meter" in meta:
                        self.pixels_per_meter = meta["pixels_per_meter"]

                # Load Annotations if CSV exists
                if csv_file:
                    df = pd.read_csv(csv_file)
                    for _, row in df.iterrows():
                        self.annotations[int(row['frame'])] = {
                            "mode": row['mode'],
                            "x": int(row['x']),
                            "y": int(row['y']),
                            "w": int(row['width']),
                            "h": int(row['height'])
                        }
                    
                    # Jump to last annotated frame
                    if self.annotations:
                        last_frame = max(self.annotations.keys())
                        self.current_frame = last_frame
                
                self.update_ui_state(has_video=True)
                self.show_frame()
                self.update_status()
                self.status_bar.showMessage(f"Project loaded from {project_dir}", 5000)
            else:
                QMessageBox.critical(self, "Error", "Could not load video file.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load project: {str(e)}")

def main():
    app = QApplication(sys.argv)
    
    # Optional: Set a dark theme style
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
