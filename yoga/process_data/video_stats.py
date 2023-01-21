from pathlib import Path
import pandas as pd
import cv2

from yoga import definitions

lookup_table = {'bhuj': 'cobra',
                'bhu': 'cobra',
                'bhuj2': 'cobra',
                'bhujan': 'cobra',
                'bhujang': 'cobra',
                'padam': 'lotus',
                'padmasan': 'lotus',
                'shav': 'corpse',
                'shava': 'corpse',
                'savasan': 'corpse',
                'shavasana': 'corpse',
                'tada': 'mountain',
                'tad': 'mountain',
                'tadasana': 'mountain',
                'tadasan': 'mountain',
                'tadasna': 'mountain',
                'trik': 'triangle',
                'trikon': 'triangle',
                'trikonasana': 'triangle',
                'vriksh': 'tree'}


def get_video_time(video_path: Path):
    """
    How many seconds does a video last.
    """
    # Read video
    video_cap = cv2.VideoCapture(str(video_path))
    # Get some video parameters
    video_n_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_fps = video_cap.get(cv2.CAP_PROP_FPS)
    video_time = round(video_n_frames / video_fps, 0)
    assert round(video_fps, 0) == 30
    return video_time


def video_df(folder_path: Path,
             save: bool = True,
             verbose: bool = True) -> pd.DataFrame:
    """
    Dataframe with the data of the 88 videos:
     - person
     - pose
     - time (s)
    """
    # Video list
    video_list = [p for p in folder_path.iterdir()
                  if p.suffix.lower() == '.mp4']

    # Process data
    video_time = [get_video_time(p) for p in video_list]
    video_names = [p.stem for p in video_list]
    data = [name.split('_') for name in video_names]
    df = pd.DataFrame(data)
    df.columns = ['person', 'pose']
    df['time(s)'] = video_time
    df['pose'] = df['pose'].str.lower()
    df['person'] = df['person'].str.capitalize()
    df['pose'].replace(lookup_table, inplace=True)
    assert df.pose.nunique() == 6

    #  Short analysis
    if verbose:
        print('Nº videos for each pose')
        print(df.groupby('pose').count()['person'])
        print('-------------------')
        print('Time(s) for each pose')
        print(df.groupby('pose').sum()['time(s)'])
        print('-------------------')
        print('Unique people')
        print(df.person.nunique())
        print('-------------------')
        print('People and number of videos')
        print(df.groupby('person').count())

    # Save dataframe
    if save:
        p = definitions.DATA_PROCESSED / 'video_stat'
        p.mkdir(parents=True, exist_ok=True)
        df.to_csv(p / 'video_list.csv')

    return df


if __name__ == '__main__':
    VIDEO_DF = video_df(definitions.DATA_EXTERNAL /
                        'Yoga_Vid_Collected')
