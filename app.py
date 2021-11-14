from flask import Flask,jsonify,request
from main_open_url import scanlicense
from flask_ngrok import run_with_ngrok
from yolo_detection_images_final import detectlicense
import json

# ================
# 設定檔案路徑
# 萬國碼顯示成繁體中文
# ================
app = Flask(__name__, static_url_path = "", static_folder = r"C:\Users\Student\Documents\Lproject\license_card\azureocr_project_resize")
app.config['JSON_AS_ASCII'] = False
@app.route('/')


# ================
# 開始執行每個def
# ================

def detect():
    img=request.args['image']                # 由query方式接收圖片路徑 ?image=  
    img_path=img
    detect=detectlicense(img_path)           # yolo 模組偵測駕照, 儲存擷取影像 , 回傳影像路徑
    scan = scanlicense(detect)               # OPencv 預處理  Azure ocr 辨識字元

    return json.dumps(scan,ensure_ascii=False)


run_with_ngrok(app)



if __name__ == "__main__":
    app.run()