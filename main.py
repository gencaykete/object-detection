from Detector import *

modelUrl = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"

classFile = "data/coco.names"
imagePath = "data/images/shop.png"
videoPath = "http://192.168.1.106:8080/video"
threshold = 0.5

detector = Detector()
detector.readClasses(classFile)
detector.downloadModel(modelUrl)
detector.loadModel()

detector.predictVideo(videoPath, threshold=threshold)
#detector.predictImage(imagePath, threshold=threshold)






