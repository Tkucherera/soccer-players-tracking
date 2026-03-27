import numpy as np
import cv2
from .models import ObjectStats

class HeatmapRenderer:

    def __init__(self, config: dict):
        self.pitch_length = config["pitch_length_m"]
        self.pitch_width  = config["pitch_width_m"]    # 68
        self.grid_size    = config["grid_size_m"]      # 2 or 5
        self.output_w     = config.get("render_width",  1050)
        self.output_h     = config.get("render_height",  680)

    def normalize(self, heatmap: np.ndarray) -> np.ndarray:
        """ 0-1 float array"""
        max_val = heatmap.max()
        if max_val == 0:
            return heatmap.astype(np.float32)
        return (heatmap/max_val).astype(np.float32)
    
    def to_image(self, heatmap: np.ndarray) -> np.ndarray:
        """
        return a Bgr image suitable for cv2.imwrite or the visualizer.
        Upscales the grid array to output_w X output_h then applies a 
        colour map so intensity reads intuitively 
        blue =cold
        red = hot 
        """
        normalized = self.normalize(heatmap)
        upscaled = cv2.resize(
            normalized,
            (self.output_w, self.output_h),
            interpolation=cv2.INTER_LINEAR # smooth blur between cells
        
        )
        as_uint8 = (upscaled * 255).astype(np.uint8)
        colored = cv2.applyColorMap(as_uint8, cv2.COLORMAP_JET)

        return colored
    
    def to_json(self, heatmap: np.ndarray) -> dict:
        """
        Serialisable dict for output_writer.py → your database/frontend.
        Sends the raw grid rather than the rendered image.
        """
        normalised = self.normalize(heatmap)
        rows, cols = normalised.shape
        return {
            "grid":        normalised.tolist(),   # 2D list of 0–1 floats
            "rows":        rows,
            "cols":        cols,
            "grid_size_m": self.grid_size,
            "pitch_length_m": self.pitch_length,
            
        }
    
    def overlay_on_pitch(self,
                         heatmap: np.ndarray,
                         pitch_img: np.ndarray,
                         alpha: float = 0.6) -> np.ndarray:
        """
        Blends the heatmap colour image over a pitch diagram image.
        pitch_img should be output_w × output_h BGR.
        """
        heatmap_img = self.to_image(heatmap)
        heatmap_img = cv2.resize(heatmap_img, 
                                  (pitch_img.shape[1], pitch_img.shape[0]))
        return cv2.addWeighted(heatmap_img, alpha, pitch_img, 1 - alpha, 0)

    def team_heatmap(self, 
                     stats_list: list[ObjectStats]) -> np.ndarray:
        """
        Sums individual heatmaps for all players in a team.
        Pass a filtered list — e.g. all players where stats.team == 0.
        """
        if not stats_list:
            rows = int(self.pitch_width  / self.grid_size)
            cols = int(self.pitch_length / self.grid_size)
            return np.zeros((rows, cols))

        combined = np.zeros_like(stats_list[0].heatmap, dtype=np.float32)
        for stats in stats_list:
            combined += stats.heatmap.astype(np.float32)
        return combined