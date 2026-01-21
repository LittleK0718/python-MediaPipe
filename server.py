import os
from flask import Flask, request, jsonify
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import text

app = Flask(__name__)

MODEL_PATH = 'bert_classifier.tflite'

if not os.path.exists(MODEL_PATH):
    print(f"error:not found {MODEL_PATH}")
    print("請確保bert_classifier.tflite在同一目錄下")
    exit(1)


base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = text.TextClassifierOptions(
    base_options=base_options,
    max_results=1  # 只回傳數值最高的
)

@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    try:
        data = request.json
        print(f"Request data: {data}")
        
        user_text = data.get('text', '')
        
        if not user_text:
            return jsonify({"status": "error", "message": "No text provided"}), 400

        print("正在使用 BERT 模型分析...")
        
        with text.TextClassifier.create_from_options(options) as classifier:
            classification_result = classifier.classify(user_text)

        
            if classification_result.classifications:
                top_category = classification_result.classifications[0].categories[0]
                sentiment_label = top_category.category_name # "positive" 或 "negative"
                confidence_score = top_category.score        # 範圍 0.0 ~ 1.0
            else:
                sentiment_label = "unknown"
                confidence_score = 0.0

        #這要不要看你 就是轉成負號
        score = confidence_score
        if sentiment_label == "negative":
            score = -confidence_score  # 如果是負面變成負數
        
        print(f"分析結果 -> 文字: {user_text}")
        print(f"sentiment: {sentiment_label}, score: {confidence_score:.4f}")

        response = {
            "status": "success",
            "score": score,      # 轉換後的數值-1~1
            "sentiment": sentiment_label,   # "positive" 或 "negative"
            "original_text": user_text
        }
        return jsonify(response)

    except Exception as e:
        print(f"發生錯誤: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Server 啟動中... (MediaPipe BERT 版)")
    print(f"📦 模型來源: {MODEL_PATH}")
    print("📡 請確保手機跟電腦連同一個 WiFi")
    
    app.run(host='0.0.0.0', port=5000)