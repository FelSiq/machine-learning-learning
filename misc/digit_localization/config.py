DATA_DIR = "./data"
OUTPUT_HEIGHT = 200
OUTPUT_WIDTH = 200
NUM_CELLS_HORIZ = 6
NUM_CELLS_VERT = 6
NUM_ANCHOR_BOXES = 1
NUM_CLASSES = 10
TARGET_DEPTH = NUM_ANCHOR_BOXES * (5 + NUM_CLASSES)

CELL_WIDTH = OUTPUT_WIDTH / NUM_CELLS_HORIZ
CELL_HEIGHT = OUTPUT_HEIGHT / NUM_CELLS_VERT
