# USAGE
# python MAIN.py
# python MAIN.py --input pedestrians.mp4
# python MAIN.py --input pedestrians.mp4 --output output.avi

# import the necessary packages
import config   # the file config.py 
from social_distance_detector import SD_detector
from mask_detector import M_detector
from tensorflow.keras.models import load_model
from datetime import datetime
from imutils.video import FPS
import tkinter as tk
from tkinter import *
from tkinter.ttk import *
from PIL import ImageTk, Image
from tkinter import filedialog as fd
import argparse
import cv2
import os

# ---------------------------- ARGUMENTS ---------------------------------
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
	help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
	help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,
	help="whether or not output frame should be displayed")
args = vars(ap.parse_args())

# ------------------------- MASK DETECTION --------------------------------
# load the serialized face detector model from disk
prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
weightsPath = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")

# ---------------------- SOCIAL DISTANCE DETECTION -------------------------
# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
personIdx=LABELS.index("person")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([config.MODEL_PATH, "yoloV4-tiny.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yoloV4-tiny.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# check if we are going to use GPU
if config.USE_GPU:
    # set CUDA as the preferable backend and target
    print("[INFO] setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

inputFile = ""
outputDirectory = ""


root=tk.Tk()    
root.title("MaskedMan Surveilance")
root.geometry("626x417")
root.grid_columnconfigure((0, 1, 2), weight=1)

min_distance = tk.StringVar(root)


background_image = Image.open('background.jpg')
bg_img = ImageTk.PhotoImage(background_image)
background_label = tk.Label(root, image = bg_img)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

image = Image.open('browsing.png')
image = image.resize((30,30), Image.ANTIALIAS)
my_img = ImageTk.PhotoImage(image)




# ----------------------- VIDEO STREAM -----------------------------------
# initialize the video stream and pointer to output video file.
def startMainFunction():
    min_distance_var = min_distance.get()
    min_distance_var = int(min_distance_var)
    print("[INFO] accessing video stream...")
    #vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
    vs = cv2.VideoCapture(inputFile)
    writer = None
    counter = 0

    # start the FPS counter
    fps = FPS().start()

    # loop over the frames from the video stream
    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()
        # if the frame was not grabbed, then we have reached the end of the stream
        if not grabbed:
            break

        # ----------------- CALL SOCIAL DISTANCE DETECTOR -------------------- 
        (sd_frame, sd_images) = SD_detector(net, ln, personIdx, frame,min_distance_var)

        # ------------------- CALL FACE-MASK DETECTOR ------------------------ 
        sound_flag = False
        # loop through the images of social distance violators
        for i in sd_images:
            # proceed only if the image is bigger than 1x1
            if i.shape[0] > 1 and i.shape[1] > 1:
                # get the image of mask violator's face
                SDFM_image = M_detector(i, faceNet, maskNet)

                # proceed only if SDFM_image is defined
                if SDFM_image is not None:
                    # save the image as a JPEG file
                    name = os.path.join(outputDirectory, datetime.now().strftime("%Y_%m_%d_%H_%M_%S-") + str(counter))
                    cv2.imwrite("%s.jpg" % name, SDFM_image)
                    counter += 1
                    sound_flag = True
  
        # if PLAY_ALARM and sound_flag is set, play sound
        if config.PLAY_ALARM and sound_flag:
            # aplay is an ALSA command, ALSA comes pre-installed in almost all linux distros
            # os.P_NOWAIT -> don't wait for os to complete execution, so we don't lose fps.
            os.spawnlp(os.P_NOWAIT, 'aplay', 'aplay', '-qd', '1', config.SOUND_FILE)

        # ----------------- DISPLAY/WRITE SOCIAL DISTANCE VIDEO --------------
        # check to see if the output frame should be displayed to our screen
        if args["display"] > 0:
            # show the output frame
            cv2.imshow("Frame", sd_frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        # if an output video file path has been supplied and the video
        # writer has not been initialized, do so now
        if args["output"] != "" and writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 25,
	    (sd_frame.shape[1], sd_frame.shape[0]), True)

        # if the video writer is not None, write the frame to the output video file
        if writer is not None:
            writer.write(sd_frame)

        # update the FPS counter
        fps.update()

    # stop the timer and display FPS information
    fps.stop()
    print("===========================")
    print("[INFO] Elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))

    # [Clean Up] close any open windows
    cv2.destroyAllWindows()




def browsefunc():
    global inputFile
    inputFile = fd.askopenfilename(filetypes=(("all video format", ".mp4"),
                    ("all video format", ".flv"),
                    ("all video format", ".mkv"),
                    ("all video format", ".wmv"),
                    ("all video format", ".avi")))
    ent1.insert(tk.END, inputFile) # add this

def askdir():
    global outputDirectory
    outputDirectory = fd.askdirectory()
    ent2.insert(tk.END, outputDirectory) # add this

input_file_label = tk.Label(root, text = 'Input File', font=('calibre',10, 'bold'))
input_file_label.grid(row = 0, column= 0,pady=(100,0))

ent1=tk.Entry(root,font=40)
ent1.grid(row=0,column=1,sticky="ew",pady=(100,0))

b1=tk.Button(root,image=my_img,font=40,command=browsefunc,pady=10,padx=10)
b1.grid(row=0,column=2,sticky="nw",pady=(100,0),padx=(10,0))



min_distance_label = tk.Label(root, text = 'Minimum Pixel', font=('calibre',10, 'bold'))
min_distance_entry = tk.Entry(root,textvariable = min_distance, font=('calibre',10,'normal'))
min_distance_label.grid(row=2,column=0,pady=(10,0))
min_distance_entry.grid(row=2,column=1,sticky="nw",pady=(10,0))


output_directory_label = tk.Label(root, text = 'Output Directory', font=('calibre',10, 'bold'))
output_directory_label.grid(row = 4, column= 0,pady=(10,0))


ent2=tk.Entry(root,font=40)
ent2.grid(row=4,column=1,sticky="ew",pady=(10,0))

b2=tk.Button(root,image=my_img,font=40,command=askdir,pady=10,padx=10)
b2.grid(row=4,column=2,sticky="nw",pady=(10,0),padx=(10,0))


b3=tk.Button(root,text="Run",font=70,command=startMainFunction)
b3.grid(row = 7,column=1,pady=(50,0))
root.mainloop()