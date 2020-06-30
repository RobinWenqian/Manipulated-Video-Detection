import pickle
from collections import defaultdict
import os

root = "Celeb-DF-v2"
txt_filepath = "Celeb-DF-v2/List_of_training_videos.txt"
pickle_filepath = "Celeb-DF-v2/List_of_training_videos.pkl"

def txt_pickle(txt_filepath,pickle_filepath):
    txt_list = []
    with open(txt_filepath,"r") as inf:
        for line in inf:
            id = line.split(' ')[0]
            video = line.strip().split(' ')[1]
            txt_list.append([video,id])

    with open(pickle_filepath,"wb") as outf:
        pickle.dump(txt_list,outf)

if __name__ == "__main__":
    txt_pickle(txt_filepath,pickle_filepath)
    with open(pickle_filepath,'rb') as pkl:
        data = pickle.load(pkl)
        print(data)
