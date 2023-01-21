from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd

from yoga import definitions


class FullBodyPoseEmbedder:
    """Converts 3D pose landmarks into 3D embedding."""

    def __init__(self, torso_size_multiplier=2.5):
        # Multiplier to apply to the torso to get minimal body size.
        self._torso_size_multiplier = torso_size_multiplier

        # Names of the landmarks as they appear in the prediction.
        self._landmark_names = [
            'nose',
            'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear',
            'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist',
            'left_pinky_1', 'right_pinky_1',
            'left_index_1', 'right_index_1',
            'left_thumb_2', 'right_thumb_2',
            'left_hip', 'right_hip',
            'left_knee', 'right_knee',
            'left_ankle', 'right_ankle',
            'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index',
        ]

    def __call__(self, landmarks):
        """Normalizes pose landmarks and converts to embedding

        Args:
          landmarks - NumPy array with 3D landmarks of shape (N, 3).

        Result:
          Numpy array with pose embedding of shape (M, 3) where `M` is the number of
          pairwise distances defined in `_get_pose_distance_embedding`.
        """
        assert landmarks.shape[0] == len(
            self._landmark_names), 'Unexpected number of landmarks: {}'.format(
            landmarks.shape[0])

        # Get pose landmarks.
        landmarks = np.copy(landmarks)

        # Normalize landmarks.
        landmarks = self._normalize_pose_landmarks(landmarks)

        # Get embedding.
        embedding = self._get_pose_distance_embedding(landmarks)

        return embedding

    def _normalize_pose_landmarks(self, landmarks):
        """Normalizes landmarks translation and scale."""
        landmarks = np.copy(landmarks)

        # Normalize translation.
        pose_center = self._get_pose_center(landmarks)
        landmarks -= pose_center

        # Normalize scale.
        pose_size = self._get_pose_size(landmarks, self._torso_size_multiplier)
        landmarks /= pose_size
        # Multiplication by 100 is not required, but makes it eaasier to debug.
        landmarks *= 100

        return landmarks

    def _get_pose_center(self, landmarks):
        """Calculates pose center as point between hips."""
        left_hip = landmarks[self._landmark_names.index('left_hip')]
        right_hip = landmarks[self._landmark_names.index('right_hip')]
        center = (left_hip + right_hip) * 0.5
        return center

    def _get_pose_size(self, landmarks, torso_size_multiplier):
        """Calculates pose size.

        It is the maximum of two values:
          * Torso size multiplied by `torso_size_multiplier`
          * Maximum distance from pose center to any pose landmark
        """
        # This approach uses only 2D landmarks to compute pose size.
        landmarks = landmarks[:, :2]

        # Hips center.
        left_hip = landmarks[self._landmark_names.index('left_hip')]
        right_hip = landmarks[self._landmark_names.index('right_hip')]
        hips = (left_hip + right_hip) * 0.5

        # Shoulders center.
        left_shoulder = landmarks[self._landmark_names.index('left_shoulder')]
        right_shoulder = landmarks[
            self._landmark_names.index('right_shoulder')]
        shoulders = (left_shoulder + right_shoulder) * 0.5

        # Torso size as the minimum body size.
        torso_size = np.linalg.norm(shoulders - hips)

        # Max dist to pose center.
        pose_center = self._get_pose_center(landmarks)
        max_dist = np.max(np.linalg.norm(landmarks - pose_center, axis=1))

        return max(torso_size * torso_size_multiplier, max_dist)

    def _get_pose_distance_embedding(self, landmarks):
        """Converts pose landmarks into 3D embedding.

        We use several pairwise 3D distances to form pose embedding. All distances
        include X and Y components with sign. We differnt types of pairs to cover
        different pose classes. Feel free to remove some or add new.

        Args:
          landmarks - NumPy array with 3D landmarks of shape (N, 3).

        Result:
          Numpy array with pose embedding of shape (M, 3) where `M` is the number of
          pairwise distances.
        """
        embedding = np.array([
            # One joint.
            self._get_distance(
                self._get_average_by_names(landmarks, 'left_hip', 'right_hip'),
                self._get_average_by_names(landmarks, 'left_shoulder',
                                           'right_shoulder')),

            self._get_distance_by_names(landmarks, 'left_shoulder',
                                        'left_elbow'),
            self._get_distance_by_names(landmarks, 'right_shoulder',
                                        'right_elbow'),

            self._get_distance_by_names(landmarks, 'left_elbow', 'left_wrist'),
            self._get_distance_by_names(landmarks, 'right_elbow',
                                        'right_wrist'),

            self._get_distance_by_names(landmarks, 'left_hip', 'left_knee'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_knee'),

            self._get_distance_by_names(landmarks, 'left_knee', 'left_ankle'),
            self._get_distance_by_names(landmarks, 'right_knee',
                                        'right_ankle'),

            # Two joints.
            self._get_distance_by_names(landmarks, 'left_shoulder',
                                        'left_wrist'),
            self._get_distance_by_names(landmarks, 'right_shoulder',
                                        'right_wrist'),

            self._get_distance_by_names(landmarks, 'left_hip', 'left_ankle'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_ankle'),

            # Four joints.

            self._get_distance_by_names(landmarks, 'left_hip', 'left_wrist'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_wrist'),

            # Five joints.

            self._get_distance_by_names(landmarks, 'left_shoulder',
                                        'left_ankle'),
            self._get_distance_by_names(landmarks, 'right_shoulder',
                                        'right_ankle'),

            self._get_distance_by_names(landmarks, 'left_hip', 'left_wrist'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_wrist'),

            # Cross body.
            self._get_distance_by_names(landmarks, 'left_elbow',
                                        'right_elbow'),
            self._get_distance_by_names(landmarks, 'left_knee', 'right_knee'),

            self._get_distance_by_names(landmarks, 'left_wrist',
                                        'right_wrist'),
            self._get_distance_by_names(landmarks, 'left_ankle',
                                        'right_ankle'),

            # Body bent direction.

            # self._get_distance(
            #     self._get_average_by_names(landmarks, 'left_wrist', 'left_ankle'),
            #     landmarks[self._landmark_names.index('left_hip')]),
            # self._get_distance(
            #     self._get_average_by_names(landmarks, 'right_wrist', 'right_ankle'),
            #     landmarks[self._landmark_names.index('right_hip')]),
        ])

        return embedding

    def _get_average_by_names(self, landmarks, name_from, name_to):
        lmk_from = landmarks[self._landmark_names.index(name_from)]
        lmk_to = landmarks[self._landmark_names.index(name_to)]
        return (lmk_from + lmk_to) * 0.5

    def _get_distance_by_names(self, landmarks, name_from, name_to):
        lmk_from = landmarks[self._landmark_names.index(name_from)]
        lmk_to = landmarks[self._landmark_names.index(name_to)]
        return self._get_distance(lmk_from, lmk_to)

    def _get_distance(self, lmk_from, lmk_to):
        return lmk_to - lmk_from


class PoseSample(object):

    def __init__(self, name, landmarks, class_name, embedding):
        self.name = name
        self.landmarks = landmarks
        self.class_name = class_name
        self.embedding = embedding


def load_pose_samples(pose_samples_folder,
                      pose_embedder,
                      file_extension='.csv',
                      n_landmarks=33,
                      n_dimensions=3):
    """Loads pose samples from a given folder.

    Required folder structure:
      corpse.csv
      tree.csv
      triangle.csv
      ...

    Required CSV structure:
      sample_00001,x1,y1,z1,x2,y2,z2,....
      sample_00002,x1,y1,z1,x2,y2,z2,....
      ...
    """
    # Each file in the folder represents one pose class.
    file_names = [p for p in pose_samples_folder.iterdir()
                  if p.suffix.lower() == file_extension]

    pose_samples = []
    for file_name in tqdm(file_names):
        # Use file name as pose class name.
        class_name = file_name.stem

        # Dataframe
        df = pd.read_csv(file_name, index_col=0)
        col_removed = list(range(4, 133, 4))
        df = df[df.columns[~df.columns.astype(int).isin(col_removed)]]
        for row in df.values:
            assert len(row) == n_landmarks * n_dimensions + 1, f'Wrong' \
                                                               f' number of values: {len(row)}'
            landmarks = np.array(row[1:], np.float32).reshape(
                [n_landmarks, n_dimensions])
            pose_samples.append(PoseSample(
                name=row[0],
                landmarks=landmarks,
                class_name=class_name,
                embedding=pose_embedder(landmarks),
            ))

    return pose_samples


def main(pose_folder: Path, save: bool = True):
    pose_embedder = FullBodyPoseEmbedder()
    data = load_pose_samples(pose_folder, pose_embedder)
    new_data = []
    for d in data:
        new_data.append([d.name, d.embedding.flatten(), d.class_name])

    df = pd.DataFrame(new_data, columns=['name', 1, 'class'])
    columns = [f'emb{i + 1}' for i in range(len(df.iloc[0, 1]))]
    split_df = pd.DataFrame(df.iloc[:, 1].tolist(), columns=columns)
    df = pd.concat([df.iloc[:, 0], split_df, df.iloc[:, 2]], axis=1)
    if save:
        df.to_csv(definitions.DATA_PROCESSED / 'final_dataset.csv')

    return df


if __name__ == '__main__':
    DF = main(definitions.DATA_PROCESSED / 'keypoints')
#
# import pandas as pd
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import make_pipeline
# from sklearn.neural_network import MLPClassifier
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# from matplotlib import ticker
# from sklearn.preprocessing import LabelEncoder
# import seaborn as sns
#
# from yoga import definitions
#
# pose_embedder = FullBodyPoseEmbedder()
# f = definitions.DATA_PROCESSED / 'keypoints'
# data = load_pose_samples(f, pose_embedder)
#
# new_data = []
# for d in data:
#     new_data.append([d.name, d.embedding.flatten(), d.class_name])
#
# df = pd.DataFrame(new_data)
# split_df = pd.DataFrame(df.iloc[:, 1].tolist())
# df = pd.concat([df, split_df], axis=1)
# df = df.drop(1, axis=1)
# X = df.iloc[:, 2:]
# y = df.iloc[:, 1]
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.33, random_state=764)
#
# clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
# clf.fit(X_train, y_train)
# print('train:', clf.score(X_train, y_train))
# print('test:', accuracy_score(y_test, clf.predict(X_test)))
#
# mlp = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
# print('Train MLP', mlp.score(X_train, y_train))
# print('Test MLP', mlp.score(X_test, y_test))
#
# t_sne = TSNE(n_components=2, learning_rate='auto',
#              init='random', perplexity=30, random_state=0)
#
# X_embedded = t_sne.fit_transform(X)
# le = LabelEncoder()
# y_true = le.fit_transform(y)
#
# plt.figure(),
# plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_true)
#
# fig, ax = plt.subplots(1, 1)
# sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=y,
#                 palette='tab10', ax=ax)
#
# for i, txt in enumerate(df.iloc[:, 0].tolist()):
#     txt = txt.split('.')[0]
#     ax.annotate(txt, (X_embedded[:, 0][i], X_embedded[:, 1][i]))
