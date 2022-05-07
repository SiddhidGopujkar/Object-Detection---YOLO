import numpy as np
import cv2
import matplotlib.pyplot as plt


def findobjects(outputs, image):
    ht, wt, ct = image.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > 0.5:
                w, h = int(det[2] * wt), int(det[3] * ht)
                x, y = int((det[0] * wt) - w / 2), int((det[1] * ht) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox, confs, 0.5, 0.3)  # confidencethreshold and NMSthreshold
    # print(indices)
    for i in indices:
        # i = j[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(image, f'{classnames[classIds[i]].upper()} {int(confs[i] * 100)}%',
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)


classnames = []
with open("coco.names", 'rt') as f:
    classnames = f.read().rstrip('\n').split('\n')

# print(classnames)

net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

image = cv2.imread("car2.jpg")
blob = cv2.dnn.blobFromImage(image, 1 / 255, (320, 320), [0, 0, 0], 1, crop=False)
net.setInput(blob)

layernames = net.getLayerNames()
# print(layernames)
for i in net.getUnconnectedOutLayers():
    print(i)
    outputnames = [layernames[i - 1]]
# print(outputnames)
# print(net.getUnconnectedOutLayers())

outputs = net.forward(outputnames)
# print(outputs[0].shape)
# print(outputs[1].shape)
# print(outputs[2].shape)
findobjects(outputs, image)

# cv2.imshow("Image", image)
plt.imshow(image)
plt.savefig("new.jpg")
plt.show()
