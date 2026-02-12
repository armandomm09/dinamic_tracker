import sys
import os
import cv2
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QStatusBar, QMessageBox, QFrame, QSizePolicy)
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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #222;") # Dark background for canvas

        self.current_frame_image = None # QPixmap
        self.mode = "point" # "point" or "bbox"
        
        # Drawing state
        self.drawing = False
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.current_annotation = None # Existing annotation for this frame (dict)
        
        # Scaling factors (displayed size vs actual video size)
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0

    def set_frame(self, frame_rgb, annotation=None):
        """Displays a new frame and any existing annotation."""
        self.current_annotation = annotation
        
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
                # We need to draw this in screen coordinates directly, 
                # but based on the start/end points which are already screen interactions.
                # However, drawing logic expects valid video coordinates usually? 
                # Actually simpler: draw directly in screen coords for preview.
                pen = QPen(Qt.green, 2, Qt.SolidLine)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(rect)

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
                    # For point, we just click. We can handle it on release or press.
                    # Let's handle immediate commit for point
                    self._commit_annotation(pos)
                    self.drawing = False # No drag for point

    def mouseMoveEvent(self, event):
        if self.drawing and self.mode == "bbox":
            self.end_point = event.pos()
            self.update() # Repaint for preview

    def mouseReleaseEvent(self, event):
        if self.drawing and self.mode == "bbox" and event.button() == Qt.LeftButton:
            self.drawing = False
            self.end_point = event.pos()
            
            # Normalize rect
            rect = self._get_rect_from_points(self.start_point, self.end_point)
            
            # If rect is too small, ignore
            if rect.width() > 5 and rect.height() > 5:
                # Convert screen rect to logical video coords
                video_rect = self._screen_to_video_rect(rect)
                if video_rect:
                    self.annotation_created.emit({
                        "mode": "bbox",
                        "x": video_rect[0],
                        "y": video_rect[1],
                        "w": video_rect[2],
                        "h": video_rect[3]
                    })

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

        # Add widgets to toolbar
        toolbar_layout.addWidget(self.btn_open)
        toolbar_layout.addWidget(self.btn_save_csv)
        toolbar_layout.addWidget(self.btn_export_video)
        toolbar_layout.addStretch() # Spacer
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
        self.btn_mode_toggle.setEnabled(has_video)
        self.btn_clear_frame.setEnabled(has_video)
        self.btn_clear_all.setEnabled(has_video)
        
        if not has_video:
            self.canvas.setText("No Video Loaded. Click 'Open Video'.")
            self.canvas.setStyleSheet("QLabel { color : white; font-size: 16px; background-color: #222; }")
        else:
            self.canvas.setText("")

    def toggle_mode(self):
        if self.btn_mode_toggle.isChecked():
            self.btn_mode_toggle.setText("Bounding Box")
            self.canvas.mode = "bbox"
        else:
            self.btn_mode_toggle.setText("Point")
            self.canvas.mode = "point"
        self.update_status()

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if path:
            if self.video_manager.load_video(path):
                self.current_frame = 0
                self.annotations = {}
                self.update_ui_state(has_video=True)
                self.show_frame()
                self.update_status()
            else:
                QMessageBox.critical(self, "Error", "Could not load video file.")

    def show_frame(self):
        """Gets frame from manager and updates canvas."""
        frame = self.video_manager.get_frame(self.current_frame)
        if frame is not None:
            # Check if we have an annotation for this frame
            ann = self.annotations.get(self.current_frame)
            self.canvas.set_frame(frame, annotation=ann)
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
        self.status_label.setText(f"{info} | {mode_str} | {ann_count}")

    # -------------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------------
    def save_csv(self):
        if not self.annotations:
            QMessageBox.information(self, "Info", "No annotations to save.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv)")
        if path:
            data = []
            # Sort by frame index
            for fid in sorted(self.annotations.keys()):
                ann = self.annotations[fid]
                data.append({
                    "frame": fid,
                    "mode": ann["mode"],
                    "x": ann["x"],
                    "y": ann["y"],
                    "width": ann["w"],
                    "height": ann["h"]
                })
            
            df = pd.DataFrame(data)
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

        # Prepare writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.video_manager.fps, 
                              (self.video_manager.width, self.video_manager.height))

        # We iterate all frames essentially re-reading the video
        # This might be slow but it's robust
        self.status_bar.showMessage("Exporting video... please wait.")
        QApplication.processEvents()

        # Create a temp capture to not mess with the UI one
        temp_cap = cv2.VideoCapture(self.video_manager.video_path)
        
        frame_idx = 0
        total = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while True:
            ret, frame = temp_cap.read()
            if not ret:
                break
            
            if frame_idx in self.annotations:
                ann = self.annotations[frame_idx]
                if ann['mode'] == 'point':
                    # Draw red circle
                    cv2.circle(frame, (ann['x'], ann['y']), 5, (0, 0, 255), -1)
                elif ann['mode'] == 'bbox':
                    # Draw green rect
                    x, y, w, h = ann['x'], ann['y'], ann['w'], ann['h']
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Draw frame number
            cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            out.write(frame)
            frame_idx += 1
            
            if frame_idx % 50 == 0:
                self.status_bar.showMessage(f"Exporting... {frame_idx}/{total}")
                QApplication.processEvents()

        temp_cap.release()
        out.release()
        self.status_bar.showMessage("Export complete!", 5000)
        QMessageBox.information(self, "Success", f"Video exported to {output_path}")

def main():
    app = QApplication(sys.argv)
    
    # Optional: Set a dark theme style
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
