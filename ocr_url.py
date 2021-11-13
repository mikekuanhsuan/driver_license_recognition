import requests
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from io import BytesIO
from pprint import pprint
from requests.models import Response

# ================
# 金鑰 端點 區域
# ================
key = ""
region = 'westus2'
endpoint = 'https://aienvision.cognitiveservices.azure.com/'
endpointUrl=f'{endpoint}vision/v3.0/ocr'

# ================
# 讀取圖片路徑
# ================
imageURL = "https://mrmad.com.tw/wp-content/uploads/2018/08/free-chinese-font.jpg"


# ================
# ocr 必要參數
# ================
headers = {
    # Request headers
    'Content-Type': 'application/json',
    'Ocp-Apim-Subscription-Key': key,
}
data = {
    'url':imageURL
}
params = {
    # Request parameters
    'language': 'zh-Hant',
    'detectOrientation': 'true',
}

# ================
# response 用json 讀取辨識的字元資料
# ================
response = requests.post(endpointUrl, headers=headers,params=params,json=data)
analysis=response.json()
pprint(analysis)

# ================
# 行列式
# ================
line_infos = [region["lines"] for region in analysis["regions"]]
pprint(analysis)





