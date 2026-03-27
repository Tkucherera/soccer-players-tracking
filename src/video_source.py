"""
VideoSource class for reading video frames and metadata using OpenCV.

This module defines the VideoSource class, which provides an iterator interface 
to read video frames along with their metadata. It uses OpenCV to handle video 
capture and supports both file paths and camera indices as sources.


Format supported: 
    - Video files (e.g., .mp4, .avi)
    - Camera devices (e.g., /dev/video0 on Linux, or index 0
    - Streaming URLs (e.g., RTSP, HTTP) if OpenCV is built with the appropriate 
      backend


"""



import cv2
import logging
from dataclasses import dataclass
from .models import FrameMetadata

logging.basicConfig(level=logging.INFO)



class VideoSource:
    def __init__(self, source: str | int, config: dict):
        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            logging.error(f"Failed to open video source: {source}")
            raise RuntimeError(f"Failed to open video source: {source}")
        logging.info(f"Video source opened: {source}")

        self.fps = self.capture.get(cv2.CAP_PROP_FPS) or config.get("default_fps", 30)
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame_number = 0
        logging.info(f"Video properties - FPS: {self.fps}, Width: {self.width}, Height: {self.height}")

    def __iter__(self):
        return self
    
    def __next__(self):
        ret, frame = self.capture.read()
        if not ret:
            logging.info("End of video stream reached")
            self.capture.release()
            raise StopIteration
        
        meta = FrameMetadata(
            frame_number=self._frame_number,
            timestamp_s=self._frame_number / self.fps,
            width=self.width,
            height=self.height
        )
        self._frame_number += 1
        return frame, meta
    
    def reset(self):                           
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._frame_number = 0
    
    def release(self):
        self.capture.release()


