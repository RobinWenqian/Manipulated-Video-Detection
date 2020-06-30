import os

root = "Celeb-DF-v2"
test_vid_file = "List_of_testing_videos.txt"
train_vid_file = "List_of_training_videos.txt"

all_video = []
test_video = []

for root, dirs, vids in os.walk(root, topdown=False):
    for video in vids:
        if len(video.split("_")) == 3 and os.path.splitext(video)[-1] == ".mp4":
            id = str(0)
            video_path = os.path.join("Celeb-synthesis",video)
            all_video.append([id,video_path])

        elif len(video.split("_")) == 2 and os.path.splitext(video)[-1] == ".mp4":
            id = str(1)
            video_path = os.path.join("Celeb-real",video)
            all_video.append([id,video_path])

with open(os.path.join(root,test_vid_file),"r") as inf:
    for line in inf:
        test_video.append([line.split(" ")[0],line.strip().split(" ")[1]])

print(len(all_video))

video_train = [x for x in all_video if x not in test_video]

print(len(video_train))

with open(os.path.join(root,train_vid_file),"w") as outf:
    for vid in video_train:
        outf.write(str(vid[0])+" "+vid[1]+"\n")
