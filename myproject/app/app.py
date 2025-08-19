from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# 아래 부분 제거 또는 주석 처리
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)