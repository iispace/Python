#%%
from ast import IsNot
import cv2
import argparse
import os


#%%
parser = argparse.ArgumentParser()
parser.add_argument('--dev_id', default=0, type=int, help='camera device id you want to use')  # Optional argument
parser.add_argument('--fps', default=30, type=int, help="Frame Per Second")                    # Optional argument
parser.add_argument('--output', type=str, help="file name to save without extension")          # Optional argument, VideoCapture will not be saved if this arg is not provided.


if __name__ == '__main__':
    args = parser.parse_args()

dev_id = args.dev_id
FPS = args.fps
filename = args.output
if filename != None and len(filename.split('.')) > 1:
    filename = filename.split('.')[0]

print(f'dev_id: {dev_id}, fps: {FPS}')
#%%
#dev_id=1
#FPS = 30.0

cap = cv2.VideoCapture(dev_id)
#cap.set(3,640)   # The first arguments is the propertyID of the video ranges from 0~18, the second argument is to set Width.

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = None
if filename != None:
    filename = filename + ".mp4"
    file_path = os.path.join(os.getcwd(), filename)
    output = cv2.VideoWriter(file_path, fourcc, FPS, (640,480))

while(True):
    ret, frame = cap.read()
    if (ret == True):
        cv2.imshow('frame', frame)
        if output != None:
            output.write(frame)
        # Close and break the loop after pressing "x" key
        if cv2.waitKey(1) &0xff == ord('x'):
            break
    else:
        print("VideoCapture Failed. Check if your camera device is on.")
        break

# close the opened camera
cap.release()
# close the opened file
if output != None:
    output.release()
# close the window and de-allocate any associated memory usage
cv2.destroyAllWindows()

# %%
