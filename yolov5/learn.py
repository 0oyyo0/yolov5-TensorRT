from utils.autoanchor import *




if __name__ == "__main__":
    kmean_anchors(dataset='/home/yy/project/yolov5/data/car_wheel.yaml', n=9, img_size=640, thr=4.0, gen=1000, verbose=True)
