import cv2
import numpy as np
import os
from tqdm import tqdm

def crop_and_speed_up_video(input_path, output_folder, start_sec, output_duration, speed_up_factor, fps=None):
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' does not exist.")
        return

    print(f"Processing video: {input_path}")

    # Open the video file
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return

    # Get video properties
    if fps is None or fps <= 0:
        fps = cap.get(cv2.CAP_PROP_FPS)
    fps = round(fps)
    if fps <= 0:
        print(f"Error: Invalid FPS value {fps}.")
        cap.release()
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video FPS: {fps}, Width: {width}, Height: {height}")

    # Calculate end_sec based on start_sec, output_duration, and speed_up_factor
    end_sec = start_sec + (output_duration * speed_up_factor)


    # Convert start and end times from seconds to frames
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    print(f"Start frame: {start_frame}, End frame: {end_frame}, FPS: {fps}")

    # Get the original video filename without extension
    original_filename = os.path.splitext(os.path.basename(input_path))[0]

    # Create output video filename
    output_filename = f"{original_filename}_s{start_sec}s_{speed_up}x_{int(output_duration)}s.mp4"
    output_path = os.path.join(output_folder, output_filename)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Set the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    total_frames = end_frame - start_frame
    with tqdm(total=total_frames, desc="Processing video", unit="frame") as pbar:
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_count >= total_frames:
                break

            # Only write every nth frame to speed up the video
            if frame_count % speed_up_factor == 0:
                out.write(frame)

            frame_count += 1
            pbar.update(1)

    # Release everything
    cap.release()
    out.release()

    print(f"Output video saved to: {output_path}")

# Example usage
input_video = os.path.abspath('RAW_SHORT_VIDEOS/camera-d8accaf0-2a04-4e83-a43c-511ec5fc0a0e_20250829_0900_s0s_1x_300s.mp4')
output_folder = 'RAW_SHORT_VIDEOS'
start_sec = 126  # Starting time in seconds for cropping
output_duration = 60  # Desired duration of the output video in seconds
speed_up = 1    # Speed up factor

# input_folder = ''

print("Starting video processing...")

# for video in input_folder:
    # input_video = os.path.join(input_folder, video)
print(f"Input video path: {input_video}")
crop_and_speed_up_video(input_video, output_folder, start_sec, output_duration, speed_up)
print("Video processing completed.")
