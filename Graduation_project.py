from ultralytics import YOLO
import cv2
from paddleocr import PaddleOCR 
import requests
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

car_plate_detector = YOLO("Plate_detecor_runs/detect/train/weights/best.pt")

# vid = cv2.VideoCapture('http://192.168.137.126/cam-mid.jpg')

vid = cv2.VideoCapture(0)

backendEndPoint = "http://www.Parking.somee.com/api/Users/Users"

MyApi = "https://abdulrahman0700-flaskapi.hf.space/verify"

plate_detection = []

Backend_plate_Num = []

counter = 0

def Backend_API(backendEndPoint,detected_license_plate,api):

    backend_response = requests.get(backendEndPoint)
    
    for Users in backend_response.json() :
        if Users['carNumber'] in detected_license_plate :
            print("Lincese are maching")
            return Open_api_control(api)
        
    print("Lincese are NOT maching")
    return close_api_control(api)
            


def Open_api_control(api):
    
    url = api  
    
    data = {'verification': 1}  
    
    response = requests.post(url, json=data)
    
    print("Response Status Code:", response.status_code)
    
    try:
        response_json = response.json()
        print("Response JSON:", response_json)
    except ValueError:
        print("Response is not in JSON format")



def close_api_control(api):

    url = api 

    data = {'verification': 0}  
    
    response = requests.post(url, json=data)
    
    print("Response Status Code:", response.status_code)
    
    try:
        response_json = response.json()
        print("Response JSON:", response_json)
    except ValueError:
        print("Response is not in JSON format")




def Paddle_OCR_Model(image): 
    ocr = PaddleOCR(use_angle_cls=True,lang='en')
    results = ocr.ocr(image,cls=True)
    license_plate = ''
    for result in results :
        for words in result :
            if isinstance(words[1][0],str):
                license_plate += words[1][0] + " "
    

    return license_plate

while vid.isOpened():
    sucess , frame = vid.read()
    if sucess :
        results = car_plate_detector(frame)[0]
        annotated_fram = results.plot()
        if results.boxes.data.tolist() :
            for res in results.boxes.data.tolist():
                x1 , y1 , x2 , y2 , score , car_id = res
                plate_detection = annotated_fram[int(y1):int(y2),int(x1):int(x2)]
                cv2.imshow('License_Plate',plate_detection)
                try :
                    
                    cv2.imwrite("cars_data/"+str(counter)+".jpg",plate_detection)
                    cv2.imshow("Result",plate_detection)
                    cv2.waitKey(500)
                    counter+=1
                    plate_num = Paddle_OCR_Model(plate_detection)
                    print(f" \n \n Licence PLate Number : {plate_num} \n \n ")

                    print(Backend_API(backendEndPoint,plate_num,MyApi))

                except :

                    print("Error in Detecting plate Licence")

        fram_resize = cv2.resize(annotated_fram,(800,500))
        cv2.imshow('My Video',fram_resize)

        if  cv2.waitKey(1) & 0xFF == ord('s'):
            break
