# base path to YOLO directory
MODEL_PATH = "yoloV4-tiny-coco"

# minimum probability to filter weak detections
MIN_CONF = 0.3

# the threshold when applying non-maxima suppression
NMS_THRESH = 0.3

# boolean indicating if NVIDIA CUDA GPU should be used
USE_GPU = False

# define the minimum safe distance (in pixels) that two people can be from each other
MIN_DISTANCE = 100

# boolean indicating if alarm has to be palyed on violations of SD&FM
PLAY_ALARM = True

# path to sound file
SOUND_FILE = "beep.wav"
