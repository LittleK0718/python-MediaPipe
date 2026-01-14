from flask import Flask, request, jsonify
from textblob import TextBlob

app = Flask(__name__)

# 設定 API 路徑
@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    try:
        # 1. 接收手機傳來的 JSON 資料
        data = request.json
        print(f"收到手機傳來的資料: {data}")
        
        user_text = data.get('text', '')
        
        # 如果沒收到文字，回傳錯誤
        if not user_text:
            return jsonify({"status": "error", "message": "No text provided"}), 400

        # 2. 進行情緒分析 (使用 TextBlob - 救急用！)
        # Polarity (極性): -1.0 (非常負面) 到 1.0 (非常正面)
        blob = TextBlob(user_text)
        score = blob.sentiment.polarity
        
        # 簡單判定情緒類別 (這裡的閥值 0.1 可以自己調整)
        sentiment_label = "neutral"
        if score > 0.1:
            sentiment_label = "positive"
        elif score < -0.1:
            sentiment_label = "negative"

        print(f"分析結果 -> 文字: {user_text}, 分數: {score}, 類別: {sentiment_label}")

        # 3. 回傳 JSON 給手機 (格式跟 MediaPipe 一模一樣，騙過手機)
        response = {
            "status": "success",
            "score": score,               # 數值 (-1 ~ 1)
            "sentiment": sentiment_label, # 文字 (positive/negative)
            "original_text": user_text
        }
        return jsonify(response)

    except Exception as e:
        print(f"發生錯誤: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # host='0.0.0.0' 代表允許區網內的手機連線
    # port=5000 是連接埠
    print("Server 啟動中... (TextBlob 救急版)")
    print("請確保手機跟電腦連同一個 WiFi")
    app.run(host='0.0.0.0', port=5000)