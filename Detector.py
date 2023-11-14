import cv2
import time
import os
import tensorflow as tf
import numpy as np

from tensorflow.python.keras.utils.data_utils import get_file

np.random.seed(123)

class Detector:
    def __init__(self):
        self.model = None
        self.cacheDir = None
        self.classesList = None
        self.colorList = None
        self.modelName = None

    def __int__(self):
        pass

    # Nesnelerin Labellarının tutulduğu dosyadan labelları aldık
    def readClasses(self, classesFilePath):
        with open(classesFilePath, 'r') as f:
            self.classesList = f.read().splitlines()

            self.colorList = np.random.uniform(low=0, high=255, size=len(self.classesList))

    # Tensorflow modellerini indirmek için
    def downloadModel(self, modelUrl):
        filename = os.path.basename(modelUrl)
        self.modelName = filename[:filename.index('.')]

        self.cacheDir = "./pretrained_models"

        os.makedirs(self.cacheDir, exist_ok=True)

        get_file(fname=filename, origin=modelUrl, cache_dir=self.cacheDir, cache_subdir="checkpoints", extract=True)

    # İndirdiğimiz modeli Yüklüyoruz
    def loadModel(self):
        print("Yüklenen Model: " + self.modelName)
        tf.keras.backend.clear_session()
        self.model = tf.saved_model.load(os.path.join(self.cacheDir, "checkpoints", self.modelName, "saved_model"))
        print("Model" + self.modelName + " yüklendi..")

    # Tanıma işleminden sonra nesneleri işaretliyoruz
    def createBoundingBox(self, image, threshold=0.5):
        inputTensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8)
        inputTensor = inputTensor[tf.newaxis, ...]

        detections = self.model(inputTensor)

        bboxs = detections['detection_boxes'][0].numpy()
        classIndexes = detections['detection_classes'][0].numpy().astype(np.int32) - 1
        classScores = detections['detection_scores'][0].numpy()

        imH, imW, imC = image.shape

        bboxIdx = tf.image.non_max_suppression(
            bboxs,
            classScores,
            max_output_size=50,
            iou_threshold=threshold,
            score_threshold=threshold
        )

        if len(bboxIdx) != 0:
            for i in bboxIdx:
                bbox = tuple(bboxs[i].tolist())
                classConfidence = round(100 * classScores[i])
                classIndex = classIndexes[i]

                classLabelText = self.classesList[classIndex]
                classColor = self.colorList[i]

                if classConfidence < 50:
                    continue

                displayText = '{}: {}%'.format(classLabelText, classConfidence)

                y_min, x_min, y_max, x_max = bbox

                x_min, x_max, y_min, y_max = (x_min * imW, x_max * imW, y_min * imH, y_max * imH)
                x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(x_max)

                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=classColor, thickness=1)
                cv2.putText(image, displayText, (x_min, y_min), cv2.FONT_HERSHEY_PLAIN, 2, classColor, 2)
        return image

    # Resimden nesne tanıma
    def predictImage(self, imagePath, threshold=0.5):
        image = cv2.imread(imagePath)

        scale_percent = 60  # percent of original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)

        # resize image
        #image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        bboxImage = self.createBoundingBox(image, threshold)

        print("Tanıma işlemi tamamlandı..")
        cv2.imshow("Sonuc", bboxImage)
        cv2.waitKey(0)

    # Videodan nesne tanıma
    def predictVideo(self, videoPath, threshold=0.5):
        cap = cv2.VideoCapture(videoPath)

        if cap.isOpened() == False:
            print("Video acilamadi ...")
            return

        (success, image) = cap.read()

        startTime = 0

        result = cv2.VideoWriter('filename.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10,
                                 (int(cap.get(3)), int(cap.get(4))))

        while success:
            currentTime = time.time()
            fps = 1 / (currentTime - startTime)
            startTime = currentTime

            bboxImage = self.createBoundingBox(image, threshold)
            cv2.putText(bboxImage, "FPS: " + str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            cv2.imshow("Sonuc", bboxImage)
            result.write(image)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            (success, image) = cap.read()

        result.release()
        cv2.destroyAllWindows()
