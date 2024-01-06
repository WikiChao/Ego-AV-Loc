from pydub import AudioSegment,silence
import csv
import os, glob
import pandas as pd

def return_dBFS(folder):
    path = glob.glob(os.path.join(folder, 'audio', '*.wav'))
    if len(path) == 0:
        return False
    myaudio = AudioSegment.from_wav(path[0])
    dBFS=myaudio.dBFS
    if dBFS > -30: # you can try different silence thresholds [-40, -30, -20, ...]
        return True
    else:
        return False

if __name__=='__main__':
    data_root = 'YOUR_DIR'
    data_dir = 'YOUR_DATA_FILE'
    list_sample = []
    for row in csv.reader(open(data_dir, 'r')):
        name = row[-1]
        dBFS = return_dBFS(name)
        if dBFS:
            print(name)
            list_sample.append(row)

    column_names = ['participant_id','video_id','start_timestamp','stop_timestamp','start_frame','stop_frame','folder_dir']

    df_train = pd.DataFrame(list_sample, columns=column_names)
    df_train.to_csv ('YOUR_SAVE_DIR', index = False, header=True)
