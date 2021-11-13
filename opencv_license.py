from collections import namedtuple
from matplotlib import pyplot as plt
import numpy as np
import argparse
import imutils
import cv2
import os


# 圖片預處理/輪廓/去背
def find_license(path):

    # ================
    # 讀取圖檔/制定圖片尺寸
    # ================
    img = cv2.imread(path)
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(1200,907))

    # ================
    # kmeans 像素分類
    # ================
    twoDimage = img.reshape((-1,3))
    twoDimage = np.float32(twoDimage)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    attempts=10
    ret,label,center=cv2.kmeans(twoDimage,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))

    # ================
    # 輪廓檢測
    # ================
    gray = cv2.cvtColor(result_image,cv2.COLOR_RGB2GRAY)
    _,thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)
    edges = cv2.dilate(cv2.Canny(thresh,0,255),None)
    cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]

    # ================
    # 製作mask
    # ================
    mask = np.zeros((907,1200), np.uint8)
    masked = cv2.drawContours(mask, [cnt],-1, 255, -1)
    dst = cv2.bitwise_and(img, img, mask=mask)
    segmented = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

    # ================
    # mask 駕照最外圍 cropped image  
    # ================
    (x,y) = np.where(mask==255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = img[x1:x2+1, y1:y2+1]    
    return cropped_image



# 分割要辨識姓名部分
def cut_license_name(img):
    # 放大尺寸 / 灰階
    cropped_image = cv2.resize(img,(4032,3024),interpolation=cv2.INTER_CUBIC)
    cropped_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # 制定長寬比例
    length = cropped_image.shape[1]
    width = cropped_image.shape[0]
    user_id_width_s = round(width*0.2)
    user_id_width_e = round(width*0.9)
    user_id_length_s = round(length*0)
    user_id_length_e = round(length*0.8)
    # 切割
    user_id = cropped_image[user_id_width_s:user_id_width_e, user_id_length_s:user_id_length_e]
    # user_id = cv2.threshold(user_id, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    return user_id


# 分割要辨識地址部分
def cut_license_address(img):
    # 放大尺寸 / 灰階
    cropped_image = cv2.resize(img,(4032,3024),interpolation=cv2.INTER_CUBIC)
    cropped_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # 制定長寬比例
    length = cropped_image.shape[1]
    width = cropped_image.shape[0]
    user_id_width_s = round(width*0.33)
    user_id_width_e = round(width*0.98)
    user_id_length_s = round(length*0)
    user_id_length_e = round(length*1.2)
    # 切割
    user_id = cropped_image[user_id_width_s:user_id_width_e, user_id_length_s:user_id_length_e]
    # user_id = cv2.threshold(user_id, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return user_id


# 分割要辨識出生部分
def cut_license_birthdate(img):
    # 放大尺寸 / 灰階
    cropped_image = cv2.resize(img,(4032,3024),interpolation=cv2.INTER_CUBIC)
    cropped_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # 制定長寬比例
    length = cropped_image.shape[1]
    width = cropped_image.shape[0]
    user_id_width_s = round(width*0.28)
    user_id_width_e = round(width*0.68)
    user_id_length_s = round(length*0.02)
    user_id_length_e = round(length*0.7)
    # 切割
    user_id = cropped_image[user_id_width_s:user_id_width_e, user_id_length_s:user_id_length_e]
    # user_id = cv2.threshold(user_id, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return user_id


# 分割要辨識ID部分
def cut_license_userid(img):
    # 放大尺寸 / 灰階
    cropped_image = cv2.resize(img,(4032,3024),interpolation=cv2.INTER_CUBIC)
    cropped_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # 制定長寬比例
    length = cropped_image.shape[1]
    width = cropped_image.shape[0]
    user_id_width_s = round(width*0)
    user_id_width_e = round(width*0.5)
    user_id_length_s = round(length*0.01)
    user_id_length_e = round(length*0.7)
    # 切割
    user_id = cropped_image[user_id_width_s:user_id_width_e, user_id_length_s:user_id_length_e]
    # user_id = cv2.threshold(user_id, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return user_id





# ================
# 測試區 
# ================
if __name__=="__main__":

    xxx = find_license("./images/detect/detectimage.jpg")
    bbb = cut_license_address(xxx)

    # 儲存圖片名稱
    output_address="output_address.jpg"
    output_id="output_id.jpg"
    output_birth="output_birth.jpg"
    output_name="output_name.jpg"
    # 儲存路徑
    img_file = "./cute_image"

    # 如果沒有資料夾路徑 幫他生一個
    if not os.path.exists(img_file):
        os.mkdir(img_file)
    # 儲存圖片
    cv2.imwrite(img_file+"/"+output_address, bbb)