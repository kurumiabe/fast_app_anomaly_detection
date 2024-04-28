# main.py
# FastAPIを用いてバックエンドAPIの処理を定義しています。
#アップロードされたZIPファイルを解凍し、画像ファイルごとに異常検知を行い、結果を返します
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from typing import List
import uvicorn
import shutil
import os
from tempfile import TemporaryDirectory
import logging
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from model import CustomModel, preprocess_image, generate_heatmap, calculate_anomaly_score,infer_and_save_heatmap
from fastapi.staticfiles import StaticFiles  # 静的ファイルのインポート

app = FastAPI()

# 静的ファイルのマウント
app.mount("/static", StaticFiles(directory="static"), name="static")

# モデルの読み込み
MODEL_PATH = "model/hz_model_ver1.pth"
model = CustomModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# 異常閾値
threshold = 0.05

@app.post("/upload_zip/") #ZIPファイルを受け取り、解凍後の画像に対して異常検知を行うエンドポイント
async def upload_zip(file: UploadFile = File(...), request: Request):
    base_url = str(request.base_url)
    if file.content_type != 'application/zip':
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a ZIP file.")
    
    with TemporaryDirectory() as tmp_dir:
        zip_path = os.path.join(tmp_dir, file.filename)
        with open(zip_path, 'wb') as f:
            shutil.copyfileobj(file.file, f)
        shutil.unpack_archive(zip_path, tmp_dir)

        results = []
        for root, _, files in os.walk(tmp_dir):
            for filename in files:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(root, filename)
                    image_tensor = preprocess_image(img_path)#画像を前処理し、テンソルに変換します。
                    
                    with torch.no_grad():
                        output = model(image_tensor)#前処理された画像テンソルをモデルに入力し、異常検知の推論を行
                    
                    anomaly_score = calculate_anomaly_score(image_tensor, output)#異常スコアを計算　オリジナルの画像テンソルと再構成された画像テンソルの間のMSEを計算し、異常スコアを返す
                    is_anomaly = anomaly_score > threshold ## 異常閾値 threshold = 0.05 を超えていたら異常
                    
                    idx = os.path.splitext(filename)[0]
                    heatmap_path = generate_heatmap(model, image_tensor, output, idx, "static") #異常があった場合、ヒートマップを生成

　　　　　　　　　　　# ベースURLの取得
　　　　　　　　　　　base_url = str(request.base_url)

　　　　　　　　　　　# ヒートマップのファイル名を決定
　　　　　　　　　　　heatmap_filename = "some_generated_name.jpg"
　　　　　　　　　　　# フルURLの生成
　　　　　　　　　　　heatmap_path = base_url + "static/" + heatmap_filename

　　　　　　　　　　　# 結果の返却
　　　　　　　　　　　results.append({
　　　　　　　　　　　"filename": filename,
　　　　　　　　　　　"anomaly_score": anomaly_score,
　　　　　　　　　　　"is_anomaly": is_anomaly,
　　　　　　　　　　　"heatmap_path": heatmap_path  # 修正されたフルURL
　　　　　　　　　　　})

        return {"results": results}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # 環境変数PORTを取得、デフォルトは8000
    uvicorn.run(app, host="0.0.0.0", port=port)
