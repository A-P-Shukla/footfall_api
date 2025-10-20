# =====================================================================================
# AI Assignment: Footfall Counter using Computer Vision
#
# counter_logic.py
#
# Description:
# This module contains the core computer vision logic for the Footfall Counter. It's
# designed to be called by an external script or API, taking a video path as input
# and returning the analysis results. It runs in a "headless" mode without GUI display.
#
# Author: Akhand Pratap Shukla
# Date: 21/10/2025
# =====================================================================================

import cv2
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO
from scipy.spatial.distance import cdist
import os

# --- Constants (can be moved to a config file in a larger application) ---
MODEL_NAME = 'yolov8n.pt'
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720
LINE_X_POSITION = FRAME_WIDTH // 2
MAX_TRACKING_DISTANCE = 75
MAX_FRAMES_TO_LOSE_TRACK = 20
TRACK_HISTORY_LENGTH = 30
HEATMAP_RADIUS = 15

def run_footfall_analysis(video_path: str, output_dir: str):
    """
    Analyzes a video file to count people crossing a virtual line.

    Args:
        video_path (str): The full path to the input video file.
        output_dir (str): The directory where the output video and heatmap will be saved.

    Returns:
        dict: A dictionary containing the final IN/OUT counts and paths to output files.
    """
    # --- Initialization ---
    track_history = defaultdict(lambda: deque(maxlen=TRACK_HISTORY_LENGTH))
    next_track_id = 0
    tracked_objects = {}
    in_count, out_count = 0, 0
    counted_in_ids, counted_out_ids = set(), set()
    heatmap = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.float32)
    last_good_frame = None

    # Load model
    print("Loading YOLO model for processing...")
    model = YOLO(MODEL_NAME)
    print("Model loaded.")

    # Open video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video file."}

    # Get video properties for output
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    video_basename = os.path.basename(video_path)
    output_video_path = os.path.join(output_dir, f"processed_{video_basename}")

# <<< ================================== THE FIX ================================== >>>
    # Use 'avc1' for H.264 encoding, which has much better compatibility with web browsers
    # in an MP4 container than 'mp4v' or 'XVID'.
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    # <<< ============================================================================= >>>
    
    out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (FRAME_WIDTH, FRAME_HEIGHT))
    if not out_writer.isOpened():
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("Error: VideoWriter failed to open. Check your FFMPEG installation and codec support.")
        print("The 'avc1' codec (H.264) might not be supported by your OpenCV build.")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return {"error": "Could not create the output video file. The 'avc1' codec might not be supported."}

    # --- Main Processing Loop ---
    print(f"Processing video: {video_path}")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        last_good_frame = frame.copy()

        # --- Object Detection ---
        results = model(frame, classes=[0], verbose=False)

        # --- Extract Detections ---
        current_detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            current_detections.append({"box": [x1, y1, x2, y2], "centroid": (cx, cy)})

        # --- Tracking ---
        if not tracked_objects:
            for i, det in enumerate(current_detections):
                tracked_objects[i] = {**det, "lost": 0}
                track_history[i].append(det["centroid"])
                next_track_id = len(current_detections)
        else:
            tracked_ids = list(tracked_objects.keys())
            prev_centroids = np.array([tracked_objects[tid]["centroid"] for tid in tracked_ids])
            current_centroids = np.array([det["centroid"] for det in current_detections])

            if len(current_centroids) > 0:
                dist_matrix = cdist(prev_centroids, current_centroids)
                used_cols = set()
                for row, tid in enumerate(tracked_ids):
                    if dist_matrix.shape[1] > 0:
                        best_match_idx = np.argmin(dist_matrix[row, :])
                        if dist_matrix[row, best_match_idx] < MAX_TRACKING_DISTANCE:
                            if best_match_idx not in used_cols:
                                tracked_objects[tid].update({**current_detections[best_match_idx], "lost": 0})
                                track_history[tid].append(current_detections[best_match_idx]["centroid"])
                                used_cols.add(best_match_idx)
                unmatched_indices = set(range(len(current_detections))) - used_cols
                for idx in unmatched_indices:
                    tracked_objects[next_track_id] = {**current_detections[idx], "lost": 0}
                    track_history[next_track_id].append(current_detections[idx]["centroid"])
                    next_track_id += 1

        # --- Counting & Lost Track Handling ---
        lost_ids = []
        for track_id, data in tracked_objects.items():
            if data['centroid'] not in [d['centroid'] for d in current_detections]:
                data["lost"] += 1
            if data["lost"] > MAX_FRAMES_TO_LOSE_TRACK:
                lost_ids.append(track_id)
            else:
                history = track_history[track_id]
                if len(history) > 1:
                    prev_x, curr_x = history[-2][0], history[-1][0]
                    if prev_x < LINE_X_POSITION and curr_x >= LINE_X_POSITION and track_id not in counted_in_ids:
                        in_count += 1
                        counted_in_ids.add(track_id)
                        counted_out_ids.discard(track_id)
                    elif prev_x > LINE_X_POSITION and curr_x <= LINE_X_POSITION and track_id not in counted_out_ids:
                        out_count += 1
                        counted_out_ids.add(track_id)
                        counted_in_ids.discard(track_id)

        for tid in lost_ids: del tracked_objects[tid]; del track_history[tid]
        
        # --- Visualization (on the frame to be saved) ---
        # (The drawing logic remains to create the output video)
        cv2.line(frame, (LINE_X_POSITION, 0), (LINE_X_POSITION, FRAME_HEIGHT), (255, 0, 0), 2)
        for track_id, data in tracked_objects.items():
            x1, y1, x2, y2 = data["box"]
            cv2.circle(heatmap, data["centroid"], HEATMAP_RADIUS, 1, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"IN: {in_count}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.putText(frame, f"OUT: {out_count}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        out_writer.write(frame)

    # --- Cleanup and Final Output Generation ---
    cap.release()
    out_writer.release()
    print("Video processing complete.")

    # Generate heatmap
    output_heatmap_path = ""
    if last_good_frame is not None:
        heatmap = cv2.GaussianBlur(heatmap, (11, 11), 0)
        heatmap_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        heatmap_color = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(heatmap_color, 0.6, last_good_frame, 0.4, 0)
        output_heatmap_path = os.path.join(output_dir, f"heatmap_{video_basename}.png")
        cv2.imwrite(output_heatmap_path, superimposed_img)
        print(f"Heatmap saved to {output_heatmap_path}")

    return {
        "in_count": in_count,
        "out_count": out_count,
        "processed_video_path": output_video_path,
        "heatmap_image_path": output_heatmap_path
    }