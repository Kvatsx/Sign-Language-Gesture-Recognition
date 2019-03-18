# Kaustav Vats (2016048)
# Lakshay Bansal ()
# Deepanshu Baadsha ()

import sys
import os
import cv2
from tqdm import tqdm

# No of frames extracted from each video
FrameCount = 20

def getFrames(path="./Data/lsa64_hand_videos/"):
    VideosList = os.listdir(path)
    if not os.path.exists(path + "../Frames"):
        os.mkdir(path + "../Frames")
    # print(VideosList)

    for vid in tqdm(VideosList):
        CaptureVideo = cv2.VideoCapture(path + vid)
        Filename = os.path.splitext(vid)
        Filename = Filename[0]
        LastFrame = None    # Storing Last Frame, if Videos doesn't contain desired number of frames
        count = 0
        FolderName = path + "../Frames/" + Filename + "/"
        if not os.path.exists(FolderName):
            os.mkdir(FolderName)
        while count < FrameCount:
            Ret, Frame = CaptureVideo.read()
            if not Ret:
                break
            ImageName = FolderName + "frame_" + str(count) + ".png"
            # Frame = cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)

            if not os.path.exists(ImageName):
                cv2.imwrite(ImageName, Frame)
                LastFrame = Frame
            count += 1

        while count < FrameCount:
            ImageName = FolderName + "frame_" + str(count) + ".png"            
            if not os.path.exists(ImageName):
                cv2.imwrite(ImageName, LastFrame)
            count += 1
        CaptureVideo.release()

if __name__ == "__main__":
    argv = sys.argv[1: ]
    if len(argv) > 1:
        print("Usage: python3 ReadFrame.py <Path>")
        sys.exit(1)

    if len(argv) == 1:
        getFrames(argv[0]) 
    else:
        getFrames()
