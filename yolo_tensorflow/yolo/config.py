import os
from pathlib import Path
#
# path and dataset parameter
#


#-----------------Parameters to be changed-----------------------------#
#############################################################################
PS_HOSTS  = '172.20.83.217:8897'

WORKER_HOSTS = '172.20.83.217:8898,172.20.83.218:8898'

BATCH_SIZE = 20

NUM_ENQUEUE_THREADS = 1

MUL_QUEUE_BATCH = 1.3

PROFILER_SAVE_STEP = 500

SUMMARY_SAVE_STEP = 5000

#############################################################################



BS_NT_MUL_PREFIX = "_bs"+str(BATCH_SIZE)+"nt"+str(NUM_ENQUEUE_THREADS)+"mul"+str(MUL_QUEUE_BATCH)+"s"

__file__= os.getcwd()
DATA_PATH = Path(__file__).parents[0]
#DATA_PATH = str(p)
print("+++++++++++++++++++++++++++++++++++++DATA_PATH"+str(DATA_PATH)+"+++++++++++++++++++++++++++++++++++++++++++++++")
PASCAL_PATH = os.path.join(str(DATA_PATH), 'pascal_voc')

CACHE_PATH = os.path.join(PASCAL_PATH, 'cache')

WEIGHTS_FILE = None
# WEIGHTS_FILE = os.path.join(DATA_PATH, 'weights', 'YOLO_small.ckpt')

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']

FLIPPED = True

#
# model parameter
#

IMAGE_SIZE = 448

CELL_SIZE = 7

MAX_OBJECT_NUM_PER_IMAGE = 7

BOXES_PER_CELL = 2

ALPHA = 0.1

DISP_CONSOLE = False

OBJECT_SCALE = 1.0
NOOBJECT_SCALE = 1.0
CLASS_SCALE = 2.0
COORD_SCALE = 5.0


#
# solver parameter
#


LEARNING_RATE = 0.0001

DECAY_STEPS = 30000

DECAY_RATE = 0.1

STAIRCASE = True

MAX_ITER = 15000

SUMMARY_ITER = 10

SAVE_ITER = 1000


#
# test parameter
#

THRESHOLD = 0.2

IOU_THRESHOLD = 0.5
