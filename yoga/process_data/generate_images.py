import cv2
from tqdm import tqdm

from yoga import definitions


def video_to_frames(pose_name: str, n: int = 60):
    """
    Split video into frames every n seconds.
    """
    # Paths
    pose_external_path = definitions.DATA_EXTERNAL / pose_name
    video_paths = [p for p in pose_external_path.iterdir()
                   if p.suffix.lower() == 'mp4']

    # Split video into frames
    for video_path in tqdm(video_paths,
                           total=len(video_paths),
                           desc='Processing the videos'):
        name = video_path.stem.split('_')[0].lower()
        video_cap = cv2.VideoCapture(str(video_path))
        video_n_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        success, image = video_cap.read()
        count = 0
        img_path = str(definitions.DATA_RAW / pose_name / name)
        while success:
            if count % n == 0:
                num = int(count / n)
                path_to_save = f'{img_path}_{num:03d}.png'
                cv2.imwrite(path_to_save, image)  # save frame as png file
            success, image = video_cap.read()
            count += 1
            if count > video_n_frames:
                video_cap.release()


if __name__ == '__main__':
    # For each pose split the video into images
    for POSE_NAME in definitions.poses:
        # check if folder exists
        p = definitions.DATA_RAW / POSE_NAME
        if not p.is_dir():
            p.mkdir(parents=True, exist_ok=True)
        print(POSE_NAME)
        video_to_frames(POSE_NAME)
        print('\n------')
