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


def get_video_list(folder_path: Path, extension: str = '.mp4') -> list:
    return [p for p in folder_path.iterdir() if p.suffix.lower() == extension]


def get_video_time(video_path: Path):
    # Read video
    video_cap = cv2.VideoCapture(str(video_path))
    # Get some video parameters
    video_n_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_fps = video_cap.get(cv2.CAP_PROP_FPS)
    video_time = round(video_n_frames / video_fps, 0)
    assert round(video_fps, 0) == 30
    return video_time


def create_video_df(video_list: list) -> pd.DataFrame:
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

    return df


if __name__ == '__main__':
    VIDEO_LIST = get_video_list(definitions.DATA_EXTERNAL /
                                'Yoga_Vid_Collected')
    VIDEO_DF = create_video_df(VIDEO_LIST)
    VIDEO_DF.to_csv(definitions.DATA_PROCESSED / 'data_videos.csv')
