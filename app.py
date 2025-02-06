import gradio as gr
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO
from pathlib import Path
import os
import subprocess
import torch
# ---------------------------
# Load the YOLOv8 model
# ---------------------------
model_path = Path("best.pt")
model = YOLO(model_path)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Fuse model layers if supported
if hasattr(model, "fuse"):
    model.fuse()

# ---------------------------
# Global variables for webcam control
# ---------------------------
streaming = False
cap = None

# ---------------------------
# Detection Functions
# ---------------------------

def detect_objects_image(image):
    """
    Perform object detection on an image and return an annotated image.
    """
    results = model(image, conf=0.5)
    annotated = results[0].plot()
    return annotated

def detect_objects_video(video_path, process_resolution=True, target_width=640, frame_interval=1):
    """
    Perform object detection on a video file.
    - process_resolution: if True, resize frame to target_width for faster processing.
    - frame_interval: process every nth frame.
    Process the video frame by frame, save the annotated video, and return its MP4 path.
    """
    cap_vid = cv2.VideoCapture(video_path)
    fps = cap_vid.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 25  # default FPS

    original_width = int(cap_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Use original resolution for writing output
    width, height = original_width, original_height

    # Create a temporary output video file (AVI container)
    temp_video_path = tempfile.NamedTemporaryFile(suffix=".avi", delete=False).name
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
    
    frame_count = 0
    while cap_vid.isOpened():
        ret, frame = cap_vid.read()
        if not ret:
            break
        frame_count += 1

        # If skipping frames, write the original frame
        if frame_count % frame_interval != 0:
            out.write(frame)
            continue

        # Optionally resize frame for processing speed
        if process_resolution:
            h, w = frame.shape[:2]
            scale = target_width / float(w)
            target_height = int(h * scale)
            processed_frame = cv2.resize(frame, (target_width, target_height))
        else:
            processed_frame = frame

        # Run inference
        results = model(processed_frame, conf=0.5)
        annotated = results[0].plot()

        # If we resized for processing, optionally scale the annotated frame back up
        if process_resolution and (annotated.shape[1] != original_width or annotated.shape[0] != original_height):
            annotated = cv2.resize(annotated, (original_width, original_height))
        
        out.write(annotated)
    
    cap_vid.release()
    out.release()
    
    # Convert AVI to browser-friendly MP4 with H.264 using FFmpeg
    final_video_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    ffmpeg_command = [
        "ffmpeg",
        "-y",  # overwrite if exists
        "-i", temp_video_path,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        final_video_path
    ]
    subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    os.remove(temp_video_path)
    return final_video_path

# ---------------------------
# Webcam Control Functions
# ---------------------------
def toggle_streaming(start: bool) -> str:
    """
    Update the global streaming flag.
    If start is True, real frames will be yielded.
    If start is False, the generator yields a blank frame.
    """
    global streaming
    streaming = start
    return "Streaming started" if streaming else "Streaming stopped"

def webcam_generator():
    """
    A generator that continuously yields webcam frames.
    If the global `streaming` flag is True, it opens the webcam, processes frames,
    and yields annotated frames. If False, it yields a blank image.
    """
    global streaming, cap
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    while True:
        if streaming:
            # Open the webcam if not already open
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            if ret:
                # Convert frame from BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model(frame, conf=0.5)
                annotated = results[0].plot()
                yield annotated
            else:
                yield blank_frame
        else:
            # If not streaming, release the webcam and yield a blank frame
            if cap is not None:
                cap.release()
                cap = None
            yield blank_frame

# ---------------------------
# Build the App using Blocks with Tabs
# ---------------------------
with gr.Blocks() as app:
    gr.Markdown("## Workplace Safety Detection")
    with gr.Tabs():
        
        # --- Tab 1: Image Detection ---
        with gr.TabItem("Image Detection"):
            with gr.Row():
                with gr.Column(scale=1): 
                    image_input = gr.Image(label="Upload Image")
                    detect_img_btn = gr.Button("Detect Objects")
                    image_examples = gr.Examples(
                        examples=[["./examples/image1.jpg"], ["./examples/image2.jpg"], ["./examples/image3.jpg"]],
                        inputs=[image_input],
                        label="Image Examples",
                    )
                with gr.Column(scale=1):
                    image_output = gr.Image(label="Annotated Image")
            detect_img_btn.click(fn=detect_objects_image, inputs=image_input, outputs=image_output)
        
        # --- Tab 2: Video Detection ---
        with gr.TabItem("Video Detection"):
            with gr.Row():
                with gr.Column(scale=1):  
                    video_input = gr.Video(label="Upload Video")
                    detect_vid_btn = gr.Button("Detect Objects")
                    video_examples = gr.Examples(
                        examples=[["./examples/video.mp4"]],
                        inputs=[video_input],
                        label="Video Examples",
                    )
                with gr.Column(scale=1):
                    video_output = gr.Video(label="Annotated Video")
            detect_vid_btn.click(fn=detect_objects_video, inputs=video_input, outputs=video_output)

        # --- Tab 3: Webcam Control ---
        with gr.TabItem("Webcam Live Detection"):    
            with gr.Row():
                start_btn = gr.Button("Start Webcam")
                stop_btn = gr.Button("Stop Webcam")
    
            status_text = gr.Textbox(label="Status", interactive=False)
    
            # Define only ONE webcam output
            webcam_output = gr.Image(label="Live Webcam Stream")  
    
            # Wire start/stop buttons to toggle the streaming flag
            start_btn.click(fn=lambda: toggle_streaming(True), outputs=status_text)
            stop_btn.click(fn=lambda: toggle_streaming(False), outputs=status_text)
    
            # Create only ONE live webcam interface
            webcam_live = gr.Interface(
                fn=webcam_generator,
                inputs=[],
                outputs=webcam_output,  
                live=True,
            )
    gr.Markdown("Created by Abdallah Ahmed")    

# ---------------------------
# Launch the App
# ---------------------------
if __name__ == "__main__":
    app.launch()