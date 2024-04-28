import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.Encoder = nn.Sequential(
            self.create_convblock(3, 16),
            nn.MaxPool2d((2, 2)),
            self.create_convblock(16, 32),
            nn.MaxPool2d((2, 2)),
            self.create_convblock(32, 64),
            nn.MaxPool2d((2, 2)),
            self.create_convblock(64, 128),
            nn.MaxPool2d((2, 2)),
            self.create_convblock(128, 256),
            nn.MaxPool2d((2, 2)),
            self.create_convblock(256, 512),
        )
        self.Decoder = nn.Sequential(
            self.create_deconvblock(512, 256),
            self.create_convblock(256, 256),
            self.create_deconvblock(256, 128),
            self.create_convblock(128, 128),
            self.create_deconvblock(128, 64),
            self.create_convblock(64, 64),
            self.create_deconvblock(64, 32),
            self.create_convblock(32, 32),
            self.create_deconvblock(32, 16),
            self.create_convblock(16, 16),
        )
        self.last_layer = nn.Conv2d(16, 3, 1, 1)

    def create_convblock(self, i_fn, o_fn):
        return nn.Sequential(
            nn.Conv2d(i_fn, o_fn, 3, 1, 1),
            nn.BatchNorm2d(o_fn),
            nn.ReLU(),
            nn.Conv2d(o_fn, o_fn, 3, 1, 1),
            nn.BatchNorm2d(o_fn),
            nn.ReLU()
        )

    def create_deconvblock(self, i_fn, o_fn):
        return nn.Sequential(
            nn.ConvTranspose2d(i_fn, o_fn, kernel_size=2, stride=2),
            nn.BatchNorm2d(o_fn),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.Encoder(x)
        x = self.Decoder(x)
        x = self.last_layer(x)
        return x

def load_model(model_path):
    model = CustomModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# def preprocess_image(image_path: str) -> torch.Tensor:
#   image = Image.open(image_path)
#   preprocess = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
#   ])
#   image_tensor = preprocess(image).unsqueeze(0)  # バッチ次元の追加
#   return image_tensor

def preprocess_image(image_path: str, augment: bool = False) -> torch.Tensor:
    """
    画像を前処理し、PyTorchのテンソルに変換します。
    テストやデモのために、オプションでデータ拡張を適用できます。

    Args:
        image_path (str): 画像のファイルパス。
        augment (bool): データ拡張を適用するかどうか（デフォルトはFalse）。

    Returns:
        torch.Tensor: 前処理された画像のテンソル。
    """
    # 基本的な前処理: リサイズとテンソル化
    preprocess_steps = [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]

    # データ拡張を追加する
    if augment:
        augmentation_steps = [
            transforms.RandomHorizontalFlip(),  # ランダムに水平反転
            transforms.RandomRotation(10),  # -10度から10度の範囲でランダムに回転
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 色調の変化
        ]
        preprocess_steps = augmentation_steps + preprocess_steps

    preprocess = transforms.Compose(preprocess_steps)

    image = Image.open(image_path).convert('RGB')  # RGB形式に変換
    image_tensor = preprocess(image).unsqueeze(0)  # バッチ次元の追加
    return image_tensor

def generate_heatmap(model, image_tensor, output, idx, results_folder):
    # ヒートマップ生成のプロセス
    difference = torch.abs(image_tensor - output)
    difference = difference.squeeze().numpy()
    difference = np.transpose(difference, (1, 2, 0))
    difference = np.clip(difference * 255, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(difference, cv2.COLORMAP_JET)
    
    # ディレクトリが存在しない場合は作成
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    heatmap_filename = f"{idx}_heatmap.jpg"
    heatmap_path = os.path.join(results_folder, heatmap_filename)
    cv2.imwrite(heatmap_path, heatmap)

    if os.path.exists(heatmap_path):
        print(f"Heatmap saved successfully at {heatmap_path}")
    else:
        print(f"Failed to save heatmap at {heatmap_path}")

    return heatmap_path



def calculate_anomaly_score(original, reconstructed):
    """
    オリジナルの画像テンソルと再構成された画像テンソルの間のMSEを計算し、異常スコアとして返します。
    
    Args:
        original (torch.Tensor): オリジナルの画像テンソル。
        reconstructed (torch.Tensor): 再構成された画像テンソル。
        
    Returns:
        float: 異常スコア（MSE）
    """
    mse_loss = torch.nn.functional.mse_loss(original, reconstructed, reduction='mean')
    return mse_loss.item()



def save_heatmap(heatmap, save_path):
    cv2.imwrite(save_path, heatmap)

def infer_and_save_heatmap(image_path, model_path, results_folder):
    model = load_model(model_path)
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image_tensor)
    reconstructed_image_tensor = output.squeeze(0)  # バッチ次元の削除
    # generate_heatmap関数を呼び出す際にモデルインスタンスを渡す
    heatmap_bytes = generate_heatmap(model, image_tensor, reconstructed_image_tensor) # 修正箇所
    heatmap_path = os.path.join(results_folder, os.path.basename(image_path) + '_heatmap.jpg')
    with open(heatmap_path, 'wb') as f:
        f.write(heatmap_bytes)
    return heatmap_path

# def detect_anomaly(original_tensor, reconstructed_tensor, threshold=0.1):
#     # 再構成誤差を計算
#     loss = torch.mean((original_tensor - reconstructed_tensor) ** 2)
#     # 閾値を超える場合は異常とみなす
#     is_anomaly = loss.item() > threshold
#     return is_anomaly

def detect_anomaly(image_tensor: torch.Tensor, model: nn.Module) -> tuple[bool, torch.Tensor]:
    with torch.no_grad():
        output = model(image_tensor)
    reconstructed_image_tensor = output.squeeze().cpu().detach()
    loss = torch.mean((image_tensor - reconstructed_image_tensor) ** 2)
    is_anomaly = loss.item() > threshold
    return is_anomaly, reconstructed_image_tensor


def predict_image(image_path, model_path):
    model = load_model(model_path)
    original_image_tensor = preprocess_image(image_path)
    
    with torch.no_grad():
        reconstructed_image_tensor = model(original_image_tensor)
    
    # 再構成誤差に基づいて異常を検出
    is_anomaly = detect_anomaly(original_image_tensor, reconstructed_image_tensor)
    
    heatmap = None
    if is_anomaly:
        # 異常が検出された場合、ヒートマップを生成
        heatmap = generate_heatmap(original_image_tensor, reconstructed_image_tensor)
    
    return {
        "is_anomaly": is_anomaly,
        "heatmap": heatmap
    }
