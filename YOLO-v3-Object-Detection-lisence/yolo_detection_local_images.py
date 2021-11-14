import numpy as np
import cv2
import requests
import shutil


def detectlicense(img_path):

    filename = img_path

    confidenceThreshold = 0.5
    NMSThreshold = 0.3

    modelConfiguration = './YOLO-v3-Object-Detection-lisence/cfg/yolov3.cfg'
    modelWeights = './YOLO-v3-Object-Detection-lisence/yolov3_4000.weights'

    labelsPath = './YOLO-v3-Object-Detection-lisence/coco.names'
    labels = open(labelsPath).read().strip().split('\n')

    np.random.seed(10)
    COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)


    img_file = "./images" 
    read_pic = img_file+"/"+filename
    image = cv2.imread(read_pic)
    # image=np.rot90(image)

    (H, W) = image.shape[:2]

    #Determine output layer names
    layerName = net.getLayerNames()
    layerName = [layerName[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB = True, crop = False)
    net.setInput(blob)
    layersOutputs = net.forward(layerName)

    boxes = []
    confidences = []
    classIDs = []
    #======== 找尋目標
    for output in layersOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confidenceThreshold:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY,  width, height) = box.astype('int')
                x = int(centerX - (width/2))
                y = int(centerY - (height/2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # #============ 切照片
    (xmin, ymin, xmax, ymax) = (boxes[0][0], boxes[0][1], boxes[0][0]+boxes[0][2], boxes[0][1]+boxes[0][3])
    box = image[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]

    # 切出來照片儲存路徑
    img_file = "./images/detect"
    # 切出來照片儲存名稱
    image="detectimage.jpg"
    imagepath = img_file+"/"+image
    cv2.imwrite(imagepath, box) 

    return imagepath

if __name__=="__main__":
    # img_path = "http://175.41.168.73:3000/uploads/29008f08-dd39-410e-89b8-6d717eadc1fd.jpeg"
    img_path = "001.jpeg"

    xxx = detectlicense(img_path)
    print(xxx)
    # cv2.imshow('Image', xxx)
    # cv2.waitKey(0)
