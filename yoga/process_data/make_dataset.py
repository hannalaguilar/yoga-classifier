import cv2
import pandas as pd
import numpy as np
import os
import sys
import tqdm

from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose

from yoga import definitions


class ImageBootstrap:
    """Helps to bootstrap images and filter pose samples for classification."""

    def __init__(self,
                 images_in_folder,
                 images_out_folder,
                 csvs_out_folder):
        self._images_in_folder = images_in_folder
        self._images_out_folder = images_out_folder
        self._csvs_out_folder = csvs_out_folder

        # Get list of pose classes and print image statistics.
        self._pose_class_names = sorted(
            [n for n in os.listdir(self._images_in_folder) if
             not n.startswith('.')])

    def bootstrap(self, per_pose_class_limit=None):
        """Bootstraps images in a given folder.

        Required image in folder (same use for image out folder):
          cobra/
            image_001.jpg
            image_002.jpg
            ...
          corpse/
            image_001.jpg
            image_002.jpg
            ...
          ...

        Produced CSVs out folder:
          cobra.csv
          corpse.csv

        Produced CSV structure with pose 3D landmarks:
          image_001,x1,y1,z1,x2,y2,z2,....
          image_002,x1,y1,z1,x2,y2,z2,....
        """
        # Create output folder for CVSs
        if not os.path.exists(self._csvs_out_folder):
            os.makedirs(self._csvs_out_folder)

        for pose_class_name in self._pose_class_names:
            print(f'Bootstrapping {pose_class_name}', file=sys.stderr)

            # Paths for the pose class
            images_in_folder = os.path.join(self._images_in_folder,
                                            pose_class_name)
            images_out_folder = os.path.join(self._images_out_folder,
                                             pose_class_name)
            csv_out_path = self._csvs_out_folder / f'{pose_class_name}.csv'
            if not os.path.exists(images_out_folder):
                os.makedirs(images_out_folder)

            # Get list of images.
            image_names = sorted(
                [n for n in os.listdir(images_in_folder) if
                 not n.startswith('.')])
            if per_pose_class_limit is not None:
                image_names = image_names[:per_pose_class_limit]

            # Initialize fresh pose tracker and run it
            with mp_pose.Pose(static_image_mode=True,
                              model_complexity=1,
                              min_detection_confidence=0.7) as pose_tracker:
                # Bootstrap every image.
                data = []
                for image_name in tqdm.tqdm(image_names):
                    # Load image.
                    input_frame = cv2.imread(
                        os.path.join(images_in_folder, image_name))
                    input_frame = cv2.cvtColor(input_frame,
                                               cv2.COLOR_BGR2RGB)

                    # Get landmarks
                    result = pose_tracker.process(image=input_frame)
                    pose_landmarks = result.pose_landmarks

                    # Save image with pose prediction
                    # (if pose was detected)
                    output_frame = input_frame.copy()
                    if pose_landmarks is not None:
                        mp_drawing.draw_landmarks(
                            image=output_frame,
                            landmark_list=pose_landmarks,
                            connections=mp_pose.POSE_CONNECTIONS)
                    output_frame = cv2.cvtColor(output_frame,
                                                cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(images_out_folder,
                                             image_name),
                                output_frame)

                    # Save landmarks if pose was detected.
                    if pose_landmarks is not None:
                        # Get landmarks.
                        frame_height = output_frame.shape[0]
                        frame_width = output_frame.shape[1]

                        pose_landmarks = np.array(
                            [[lmk.x * frame_width, lmk.y * frame_height,
                              lmk.z * frame_width, lmk.visibility * 100]
                             for lmk in pose_landmarks.landmark],
                            dtype=np.float32)
                        assert pose_landmarks.shape == (33, 4),\
                            f'Unexpected landmarks shape: ' \
                            f'{pose_landmarks.shape}'

                        data.append([image_name] +
                                    pose_landmarks.flatten().tolist())

                pd.DataFrame(data).to_csv(csv_out_path)

    def check_images_and_csvs(self, print_removed_items=False):
        """Makes sure that image folders and CSVs have the same sample.

        Leaves only intersection of samples in both image folders and CSVs.
        """
        for pose_class_name in self._pose_class_names:
            # Paths for the pose class
            images_out_folder = os.path.join(self._images_out_folder,
                                             pose_class_name)
            csv_out_path = os.path.join(self._csvs_out_folder,
                                        pose_class_name + '.csv')

            # Read CSV
            df = pd.read_csv(csv_out_path, index_col=0)
            # Read list of images
            image_names = sorted(
                [n for n in os.listdir(images_out_folder) if
                 not n.startswith('.')])
            # Check image and csv
            removed_items = []
            for image_name_csv, image_name in zip(df.iloc[:, 0].tolist(),
                                                  image_names):
                if not image_name_csv == image_name:
                    image_path = os.path.join(images_out_folder, image_name)
                    os.remove(image_path)
                    if print_removed_items:
                        print(f'Removed image from folder and CSV:'
                              f' {image_name}')
                    removed_items.append(image_name)
            # Delete if there are removed items
            if len(removed_items) != 0:
                df = df[~df.iloc[:, 0].isin(removed_items)]
                df.to_csv(csv_out_path)
            else:
                print(f'Nothing to remove for {pose_class_name}')

    def print_images_in_statistics(self):
        """Prints statistics from the input image folder"""
        self._print_images_statistics(self._images_in_folder,
                                      self._pose_class_names)

    def print_images_out_statistics(self):
        """Prints statistics from the output image folder"""
        self._print_images_statistics(self._images_out_folder,
                                      self._pose_class_names)

    @staticmethod
    def _print_images_statistics(images_folder, pose_class_names):
        print('Number of images per pose class:')
        for pose_class_name in pose_class_names:
            n_images = len([
                n for n in
                os.listdir(os.path.join(images_folder, pose_class_name))
                if not n.startswith('.')])
            print(f'  {pose_class_name}: {n_images}')


def main(bootstrap_images_in_folder,
         bootstrap_images_out_folder,
         bootstrap_csvs_out_folder,
         limit=None):

    # Initialize helper
    bootstrap_helper = ImageBootstrap(
        images_in_folder=bootstrap_images_in_folder,
        images_out_folder=bootstrap_images_out_folder,
        csvs_out_folder=bootstrap_csvs_out_folder,
    )
    # Check how many pose classes and images for them are available
    bootstrap_helper.print_images_in_statistics()

    # Bootstrap all images
    # Set limit to some small number for debug
    bootstrap_helper.bootstrap(per_pose_class_limit=limit)

    # Check how many images were bootstrapped
    bootstrap_helper.print_images_out_statistics()

    # After initial bootstrapping images without detected poses
    # were still saved in the folder (but not in the CSVs)
    # for debug purpose. Let's remove them
    bootstrap_helper.check_images_and_csvs(print_removed_items=True)
    bootstrap_helper.print_images_out_statistics()


if __name__ == '__main__':
    main(definitions.DATA_RAW / 'filtered_images',
         definitions.DATA_TMP / 'filtered_images',
         definitions.DATA_PROCESSED / 'keypoints',
         limit=None)
