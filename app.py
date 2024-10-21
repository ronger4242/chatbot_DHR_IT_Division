from flask import Flask, render_template, request, jsonify
from query_data import query_data, get_db_info
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(script_dir, 'templates')

app = Flask(__name__, template_folder=template_dir)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    query_text = request.json['question']
    response_text = query_data(query_text)
    return jsonify({'answer': response_text})

@app.route('/db_info', methods=['GET'])
def db_info():
    count = get_db_info()
    return jsonify({'document_count': count})

if __name__ == '__main__':
    app.run(debug=True)