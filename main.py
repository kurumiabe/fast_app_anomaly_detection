# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.staticfiles import StaticFiles
import uvicorn
import shutil
import os
from tempfile import TemporaryDirectory
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from model import CustomModel, preprocess_image, generate_heatmap, calculate_anomaly_score
from urllib.parse import urljoin

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

@app.post("/upload_zip/")
async def upload_zip(request: Request, file: UploadFile = File(...)):
    if file.content_type != 'application/zip':
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a ZIP file.")
    
    with TemporaryDirectory() as tmp_dir:
        zip_path = os.path.join(tmp_dir, file.filename)
        with open(zip_path, 'wb') as f:
            shutil.copyfileobj(file.file, f)
        shutil.unpack_archive(zip_path, tmp_dir)

        results = []
        for root, _, files in os.walk(tmp_dir):
            base_url = request.base_url.rstrip('/') + '/'  # 確実に末尾にスラッシュを保持
            for filename in files:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(root, filename)
                    image_tensor = preprocess_image(img_path)
                    
                    with torch.no_grad():
                        output = model(image_tensor)
                    
                    anomaly_score = calculate_anomaly_score(image_tensor, output)
                    is_anomaly = anomaly_score > threshold
                    heatmap_filename = f"{os.path.splitext(filename)[0]}_heatmap.jpg"
                    full_heatmap_path = urljoin(base_url, f"static/{heatmap_filename}")
                    
                    results.append({
                        "filename": filename,
                        "anomaly_score": anomaly_score,
                        "is_anomaly": is_anomaly,
                        "heatmap_path": full_heatmap_path
                    })

        return {"results": results}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
