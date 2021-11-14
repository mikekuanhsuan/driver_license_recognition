import requests
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from io import BytesIO
from pprint import pprint
from requests.models import Response
import re

def read_name(path,key,region,endpoint):

    # ================
    # Azure ocr 辨識字元
    # ================

    # 金鑰 端點 區域
    key = key
    region = region
    endpoint = endpoint
    endpointUrl=f'{endpoint}vision/v3.0/ocr'

    # 圖片路徑 / rb讀取
    image_path = path
    image_data = open(image_path, "rb").read()

    # 帶入ocr api 需要的參數
    headers = {'Ocp-Apim-Subscription-Key': key,
            'Content-Type': 'application/octet-stream'}
    params = {
        # Request parameters
        'language': 'zh-Hant',
        'detectOrientation': 'true',
    }

    # response 用json 讀取辨識的字元資料
    response = requests.post(
        endpointUrl, headers=headers, params=params, data=image_data)
    response.raise_for_status()
    analysis=response.json()
    # pprint(analysis)

    # 行列式
    line_infos = [region["lines"] for region in analysis["regions"]]
    # pprint(line_infos)

    # 行列式存成list
    word_infos = []
    for line in line_infos:
        for word_metadata in line:
            for word_info in word_metadata["words"]:
                word_infos.append(word_info)
    # print(word_infos)

    # ================
    # 制定boundingbox 的上下限
    # ================
    # 上限X
    x_start=[]                         ## 找到碼的y軸
    for wordall in word_infos:
        Box = wordall['boundingBox']
        text_address=wordall["text"]
        if '碼' in text_address:                         # 使用in運算子檢查
            x_start.append(Box)
        elif( '名' in text_address ):
            x_start.append(Box)   
        elif( '生' in text_address ):
            x_start.append(Box)
        elif( '期' in text_address ):
            x_start.append(Box)     
    x_start = x_start[0]
    x_start = x_start.split(',')
    x_start = int(x_start[0])+10 

    # 下限X
    x_end=[]                         ## 找到碼的y軸
    for wordall in word_infos:
        Box = wordall['boundingBox']
        text_address=wordall["text"]
        if '性' in text_address:                         # 使用in運算子檢查
            x_end.append(Box)
        elif( '血' in text_address ):
            x_end.append(Box)
        elif( '次' in text_address ):
            x_end.append(Box)
        elif( '補' in text_address ):
            x_end.append(Box) 
    x_end = x_end[0]
    x_end = x_end.split(',')
    x_end = int(x_end[0])-10    


    # 上限Y
    y_start=[]                         ## 找到碼的y軸
    for wordall in word_infos:
        Box = wordall['boundingBox']
        text_address=wordall["text"]
        if '碼' in text_address:                         # 使用in運算子檢查
            y_start.append(Box)
        elif( '次' in text_address ):
            y_start.append(Box) 
        elif( '數' in text_address ):
            y_start.append(Box)
        elif( '號' in text_address ):
            y_start.append(Box) 
    
    y_start = y_start[0]
    y_start = y_start.split(',')
    y_start = int(y_start[1])+10

    # 下限Y
    y_end=[]                         ## 找到效的y軸
    for wordall in word_infos:
        Box = wordall['boundingBox']
        text_address=wordall["text"]
        if '生' in text_address:                         # 使用in運算子檢查
            y_end.append(Box)
        elif( '出' in text_address ):
            y_end.append(Box)
        elif( '血' in text_address ):
            y_end.append(Box)
        elif( '型' in text_address ):
            y_end.append(Box)
    y_end = y_end[0]
    y_end = y_end.split(',')
    y_end = int(y_end[1])-10
    

    # ================
    # 重新搜尋地址並且sort
    # ================
    newbox=[]                                  # 重新搜尋地址並且sort
    for wordall in word_infos:
        text_address=wordall["text"]
        Box = wordall['boundingBox']
        Box = Box.split(',')                  # 取出來是字串 直接split
        bounding_y = int(Box[1])
        bounding_x = int(Box[0])
        if( bounding_x > x_start and bounding_x < x_end and bounding_y > y_start and bounding_y < y_end ):
            newbox.append(text_address)

    # ================
    # 每個字元接再一起 / 消除不要字元
    # ================
    address="".join(newbox)
    cleantext = re.sub('[\a-zA-Z0-9\W]+', '', address) 

    return cleantext

if __name__=="__main__":
    tryimage = "./cute_image/output_name.jpg"
    key = "d620bc8b73ad4e6aad0c28034e70adb3"
    region = 'westus2'
    endpoint = 'https://aienvision.cognitiveservices.azure.com/'

    xxx = read_name(tryimage,key,region,endpoint)
    pprint(xxx)
