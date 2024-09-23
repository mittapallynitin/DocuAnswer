import os

import dotenv
from flask import Flask, render_template, request

import service

dotenv.load_dotenv()

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/greet', methods=['GET'])
def greet_user():
    user = request.args.get('user')
    return {'greeting': f'hi {user}'}

@app.route('/upload', methods=['POST'])
def upload_pdf():
    pdf = request.files['file']
    # Process PDF and prepare for Q&A
    print("Received File")
    return service.load_pdf(pdf)

@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.form['question']
    # Retrieve relevant chunks and query OpenAI API
    return "Answer to your question."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.environ['PORT'] or 80)
