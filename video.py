from nuscenes.nuscenes import NuScenes
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import load_prediction
from nuscenes.eval.common.utils import boxes_to_sensor
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points

import os
import matplotlib.pyplot as plt
import numpy as np

import imageio.v2 as imageio   
from PIL import Image          

# Config
DATA_ROOT = "/Users/nicolas/Desktop/numini"     
VERSION = "v1.0-mini"                           
RESULTS_JSON = "/Users/nicolas/Desktop/detection/results_nusc.json"  
OUT_DIR = "./nuscenes_screenshots"              
os.makedirs(OUT_DIR, exist_ok=True)

PRED_SCORE_TH = 0.30

MAKE_VIDEO = True
FPS = 5 
CAM_VIDEO_PATH = os.path.join(OUT_DIR, "cam_front.mp4")
LIDAR_VIDEO_PATH = os.path.join(OUT_DIR, "lidar_top.mp4")


# Load nuscenes and predictions
nusc = NuScenes(version=VERSION, dataroot=DATA_ROOT, verbose=True)

pred_boxes: EvalBoxes
pred_boxes, _ = load_prediction(
    RESULTS_JSON,
    max_boxes_per_sample=500,
    box_cls=DetectionBox
)

sample_tokens = sorted(pred_boxes.sample_tokens)


# Helpers

def plot_boxes_on_image(nusc: NuScenes,
                        sample_token: str,
                        pred_boxes_for_sample,
                        out_path: str,
                        conf_th: float = 0.30) -> None:
    sample = nusc.get('sample', sample_token)
    cam_token = sample['data']['CAM_FRONT']
    cam_data = nusc.get('sample_data', cam_token)
    cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', cam_data['ego_pose_token'])

    img_path = os.path.join(nusc.dataroot, cam_data['filename'])
    img = plt.imread(img_path)

    K = np.array(cs_record['camera_intrinsic'])
    boxes_cam = boxes_to_sensor(pred_boxes_for_sample, pose_record, cs_record)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.imshow(img)
    ax.set_title(f"Sample: {sample_token}")

    for box_cam, box_global in zip(boxes_cam, pred_boxes_for_sample):
        score = getattr(box_global, "detection_score", 1.0)
        if score is None or np.isnan(score) or score < conf_th:
            continue

        corners_3d = box_cam.corners()
        corners_2d = view_points(corners_3d, K, normalize=True)

        if (corners_2d[2, :] > 0).all():
            x = corners_2d[0, :]
            y = corners_2d[1, :]

          
            for i, j in [(0, 1), (1, 2), (2, 3), (3, 0)]:
                ax.plot([x[i], x[j]], [y[i], y[j]])

      
            for i, j in [(4, 5), (5, 6), (6, 7), (7, 4)]:
                ax.plot([x[i], x[j]], [y[i], y[j]])

          
            for i in range(4):
                ax.plot([x[i], x[i + 4]], [y[i], y[i + 4]])

    ax.axis('off')
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def plot_boxes_on_lidar(nusc: NuScenes,
                        sample_token: str,
                        pred_boxes_for_sample,
                        out_path: str,
                        conf_th: float = 0.30) -> None:
    sample = nusc.get('sample', sample_token)
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)
    lidar_file = os.path.join(nusc.dataroot, lidar_data['filename'])
    cs_record = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', lidar_data['ego_pose_token'])

    pc = LidarPointCloud.from_file(lidar_file)
    boxes_lidar = boxes_to_sensor(pred_boxes_for_sample, pose_record, cs_record)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(pc.points[0, :], pc.points[1, :], s=0.5)
    ax.set_aspect('equal', 'box')
    ax.set_title(f"LiDAR TOP view: {sample_token}")

    for box_lidar, box_global in zip(boxes_lidar, pred_boxes_for_sample):
        score = getattr(box_global, "detection_score", 1.0)
        if score is None or np.isnan(score) or score < conf_th:
            continue

        corners = box_lidar.corners()
        xs = corners[0, [0, 1, 2, 3, 0]]
        ys = corners[1, [0, 1, 2, 3, 0]]
        ax.plot(xs, ys)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


# Generate screenshots


for idx, sample_token in enumerate(sample_tokens):
    boxes_sample = pred_boxes[sample_token]

    if len(boxes_sample) == 0:
        continue 

    img_out = os.path.join(OUT_DIR, f"{idx:03d}_cam_front.png")
    plot_boxes_on_image(nusc, sample_token, boxes_sample, img_out, conf_th=PRED_SCORE_TH)

    lidar_out = os.path.join(OUT_DIR, f"{idx:03d}_lidar_top.png")
    plot_boxes_on_lidar(nusc, sample_token, boxes_sample, lidar_out, conf_th=PRED_SCORE_TH)

print(f"Done. Screenshots saved in: {OUT_DIR}")


# Make videos from screenshots

def ensure_even_size(size):
    """Make width and height even for libx264."""
    w, h = size
    if w % 2 != 0:
        w -= 1
    if h % 2 != 0:
        h -= 1
    w = max(2, w)
    h = max(2, h)
    return (w, h)


def write_video(frames, video_path):
    if not frames:
        print(f"No frames found for {video_path}, skipping.")
        return

    target_size = None  

    with imageio.get_writer(video_path, fps=FPS, macro_block_size=None) as writer:
        for fname in frames:
            frame = imageio.imread(os.path.join(OUT_DIR, fname))

            if target_size is None:
                w, h = frame.shape[1], frame.shape[0]
                target_size = ensure_even_size((w, h))

            if (frame.shape[1], frame.shape[0]) != target_size:
                frame = np.array(
                    Image.fromarray(frame).resize(target_size, Image.BILINEAR)
                )

            writer.append_data(frame)

    print(f"Video saved to: {video_path}")


if MAKE_VIDEO:
    cam_frames = sorted(
        f for f in os.listdir(OUT_DIR) if f.endswith("_cam_front.png")
    )
    write_video(cam_frames, CAM_VIDEO_PATH)

    lidar_frames = sorted(
        f for f in os.listdir(OUT_DIR) if f.endswith("_lidar_top.png")
    )
    write_video(lidar_frames, LIDAR_VIDEO_PATH)
