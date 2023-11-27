import numpy as np
import cv2

imgfile = "1.png"
video_dim = (1280, 720)
fps = 25
duration = 2.0
start_center = (0.4, 0.6)
end_center = (0.5, 0.5)
start_scale = 0.7
end_scale = 1.0

img = cv2.imread(imgfile, cv2.IMREAD_COLOR)
orig_shape = img.shape[:2]


def crop(img, x, y, w, h):
    x0, y0 = max(0, x - w // 2), max(0, y - h // 2)
    x1, y1 = x0 + w, y0 + h
    return img[y0:y1, x0:x1]


num_frames = int(fps * duration)
frames = []
for alpha in np.linspace(0, 1, num_frames):
    rx = end_center[0] * alpha + start_center[0] * (1 - alpha)
    ry = end_center[1] * alpha + start_center[1] * (1 - alpha)
    x = int(orig_shape[1] * rx)
    y = int(orig_shape[0] * rx)
    scale = end_scale * alpha + start_scale * (1 - alpha)
    # determined how to crop based on the aspect ratio of width/height
    if orig_shape[1] / orig_shape[0] > video_dim[0] / video_dim[1]:
        h = int(orig_shape[0] * scale)
        w = int(h * video_dim[0] / video_dim[1])
    else:
        w = int(orig_shape[1] * scale)
        h = int(w * video_dim[1] / video_dim[0])
    # crop, scale to video size, and save the frame
    cropped = crop(img, x, y, w, h)
    scaled = cv2.resize(cropped, dsize=video_dim, interpolation=cv2.INTER_LINEAR)
    frames.append(scaled)

# write to MP4 file
vidwriter = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, video_dim)
for frame in frames:
    vidwriter.write(frame)
vidwriter.release()