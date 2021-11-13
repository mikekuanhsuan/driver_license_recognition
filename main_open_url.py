import os
from os import name
from ocr_local_address import read_address
from ocr_local_name import read_name
from ocr_local_birth import read_birth
from ocr_local_userid import read_userid
from opencv_license import find_license,cut_license_name,cut_license_address,cut_license_birthdate,cut_license_userid
import imutils
import cv2
import urllib
import urllib.request as request
import numpy as np
import requests
import shutil


def scanlicense(img_path):

    # ================
    # 讀取yolo擷取的照片路徑
    # ================
    scan_pic_path = img_path

    # ================
    # 影像kmeans顏色分類輪廓檢測
    # ================
    cropped_image=find_license(scan_pic_path)

    # ================
    # 分別切割出部分圖檔
    # ================
    cut_name = cut_license_name(cropped_image)
    cut_address = cut_license_address(cropped_image)
    cut_birthdate = cut_license_birthdate(cropped_image)
    cut_id = cut_license_userid(cropped_image)


    # ================
    # 儲存圖片
    # ================
    output_name="output_name.jpg"
    output_address="output_address.jpg"
    output_birth="output_birth.jpg"
    output_id="output_id.jpg"

    # 儲存路徑
    img_file = "./cute_image"

    # 如果沒有資料夾路徑 幫他生一個
    if not os.path.exists(img_file):
        os.mkdir(img_file)

    # 儲存圖片
    cv2.imwrite(img_file+"/"+output_name, cut_name)
    cv2.imwrite(img_file+"/"+output_address, cut_address)
    cv2.imwrite(img_file+"/"+output_birth, cut_birthdate) 
    cv2.imwrite(img_file+"/"+output_id, cut_id) 


    # ================
    # Azure ocr 辨識
    # ================

    # AZURE key endpoint
    key = ""                              ## 需要自行取得金鑰
    region = 'westus2'
    endpoint = 'https://aienvision.cognitiveservices.azure.com/'


    # 要辨識的照片路徑
    address_path = "./cute_image/output_address.jpg"
    name_path = "./cute_image/output_name.jpg"
    birth_path = "./cute_image/output_birth.jpg"
    userid_path = "./cute_image/output_id.jpg"

    # 帶入函式中處理
    address = read_address(address_path,key,region,endpoint)
    name = read_name(name_path,key,region,endpoint)
    birth = read_birth(birth_path,key,region,endpoint)
    userid = read_userid(userid_path,key,region,endpoint)

    license_information={"name":name,"add":address,"birth":birth,"ID":userid}

    return license_information

if __name__=="__main__":
    xxx = scanlicense("./images/detect/detectimage.jpg")
    print(xxx)
