import streamlit as st
import requests

st.title('胡桃の異常検知アプリ')

uploaded_file = st.file_uploader("Zipファイルをアップロードしてください", type="zip")
if uploaded_file is not None and st.button('分類開始'):#ファイルがアップロードされて,「分類開始」であるボタンが押された場合
    files = {"file": (uploaded_file.name, uploaded_file, "application/zip")}
    response = requests.post("https://fast-app-anomaly-detection.onrender.com/upload_zip/", files=files)#アップロードされたファイルをFastAPIサーバーに送信
    if response.status_code == 200:
        results = response.json()["results"]
        for result in results:
            st.write(f"ファイル名: {result['filename']}, 異常スコア: {result['anomaly_score']:.4f}")
            if result['is_anomaly']:
                st.error("異常が検出されました！")
                # ヒートマップの表示
                st.image(result['heatmap_path'], caption="Anomaly Heatmap")#異常検知の結果として得られたヒートマップを表示
            else:
                st.success("異常はありません。")
    else:
        st.error(f"エラーが発生しました。ステータスコード: {response.status_code}, メッセージ: {response.text}")
