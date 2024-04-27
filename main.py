# main.py
# FastAPIを用いてバックエンドAPIの処理を定義しています。
#アップロードされたZIPファイルを解凍し、画像ファイルごとに異常検知を行い、結果を返
from fastapi import FastAPI, File, UploadFile, HTTPException
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
from model import *

app = FastAPI()

# モデルの読み込み（適宜パスを修正してください）
MODEL_PATH = "model/hz_model_ver1.pth"
model = CustomModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# 異常閾値
threshold = 0.05

@app.post("/upload_zip/") #ZIPファイルを受け取り、解凍後の画像に対して異常検知を行うエンドポイント
async def upload_zip(file: UploadFile = File(...)):
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
                    heatmap_path = generate_heatmap(model, image_tensor, output, idx, tmp_dir)#異常があった場合、ヒートマップを生成

                    results.append({
                        "filename": filename,
                        "anomaly_score": anomaly_score,
                        "is_anomaly": is_anomaly,
                        "heatmap_path": heatmap_path
                    })
        return {"results": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# @app.post("/upload_zip/")
# async def upload_zip(file: UploadFile = File(...)):
#     if file.content_type != 'application/zip':
#         raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a ZIP file.")
    
#     with TemporaryDirectory() as tmp_dir:
#         zip_path = os.path.join(tmp_dir, file.filename)
#         with open(zip_path, 'wb') as f:
#             shutil.copyfileobj(file.file, f)
#         shutil.unpack_archive(zip_path, tmp_dir)

#         results = []
#         for root, _, files in os.walk(tmp_dir):
#             for filename in files:
#                 if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#                     img_path = os.path.join(root, filename)
#                     image_tensor = preprocess_image(img_path)  # 画像をテンソルに変換

#                     with torch.no_grad():
#                         output = model(image_tensor)  # モデルで予測

#                     # idxはファイル名など一意の識別子を使用
#                     idx = os.path.splitext(filename)[0]

#                     # 修正: generate_heatmap関数にimage_tensorも渡す
#                     heatmap_path = generate_heatmap(model, image_tensor, output, idx, tmp_dir)

                
#                 results.append({
#                     "filename": filename,
#                     "anomaly_score": anomaly_score,
#                     "is_anomaly": is_anomaly,
#                     "heatmap_path": heatmap_path  # ヒートマップのファイルパスを結果に含める
#                 })
#         return {"results": results}

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
