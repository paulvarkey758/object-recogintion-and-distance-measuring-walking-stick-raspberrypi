import cv2
import os
import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
trig=12
echo=13
GPIO.setup(echo,GPIO.IN)
GPIO.setup(trig,GPIO.OUT)

#thres = 0.45 # Threshold to detect object

classNames = []
classFile = "/home/pi/Documents/object-recognition-with-distance/file/coco.names"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "/home/pi/Documents/object-recognition-with-distance/file/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/pi/Documents/object-recognition-with-distance/file/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    #print(classIds,bbox)
    object_confidence=[]
    object_name=[]
    if len(objects) == 0: objects = classNames
    objectInfo =[]
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box,className])
                if (draw):
                    cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                    cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    object_confidence.append(round(confidence*100,2))
                    object_name.append(classNames[classId-1])

    return img,objectInfo,object_confidence,object_name

def distance():
    GPIO.output(trig,True)
    time.sleep(0.00001)
    GPIO.output(trig,False)
    pulse_start=time.time()
    pulse_end=time.time()
    while GPIO.input(echo)==0:
        pulse_start=time.time()
    while GPIO.input(echo)==1:
        pulse_end=time.time()
    pulse_duration=pulse_end-pulse_start
    distance=(pulse_duration*34300)/2
    distance=int(distance)
    return distance
    

if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)
    #cap.set(10,70)


    while True:
        s_time=time.time()
        e_time=0
        while True:
            success, img = cap.read()
            result, objectInfo,object_confidence,object_name = getObjects(img,0.45,0.2)
            # print(objectInfo)
            cv2.imshow("Output",img)
            e_time=time.time()
            cv2.waitKey(1)
            if round(e_time-s_time)>=5:
                break
        if len(object_confidence)>0:
            max_confidence=max(object_confidence)
            if max_confidence>45:
                conf_index=object_confidence.index(max_confidence)
                print(object_name[conf_index])
                os.system(f'echo {object_name[conf_index]} | festival --tts')
        dist=distance()
        print(dist)
        if dist<250:
            os.system(f'echo "distance is" {dist} "cm" | festival --tts') 
        time.sleep(.25)
        

