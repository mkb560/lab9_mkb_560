import os
import uuid
from flask import Flask, request, jsonify, render_template, session
from werkzeug.utils import secure_filename

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- Set OpenAI API Key ---
# Replace 'your-actual-api-key-here' with your real OpenAI key
# os.environ["OPENAI_API_KEY"] = "your-actual-api-key-here"

from langchain_openai import ChatOpenAI
import pdf_extractor
import vectorstore_builder
import conversation_chain

app = Flask(__name__)
app.secret_key = os.urandom(24)

UPLOAD_FOLDER = 'pdfs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global dictionary to store isolated conversation chains per user session
user_chains = {}

@app.route('/')
def index():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    """
    Handle PDF uploads, extract text, and build the vector store.
    """
    if 'files' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No selected files'}), 400

    saved_files = []
    for file in files:
        if file and file.filename.endswith('.pdf'):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            saved_files.append(file_path)

    if not saved_files:
        return jsonify({'error': 'Only PDF files are allowed'}), 400

    try:
        chunks = pdf_extractor.run_pipeline(app.config['UPLOAD_FOLDER'])
        vectorstore_builder.build_and_save_openai(chunks, 'faiss_index_openai')
        user_chains.clear()
        
        return jsonify({'message': f'Successfully uploaded and analyzed {len(saved_files)} PDF(s).'})
    except Exception as e:
        return jsonify({'error': f'An error occurred during processing: {str(e)}'}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    """
    Receive user question, pass it to the user's specific conversation chain, and return the answer.
    """
    data = request.get_json()
    user_question = data.get('question')

    if not user_question:
        return jsonify({'error': 'No question provided'}), 400

    user_id = session.get('user_id')
    if not user_id:
        user_id = str(uuid.uuid4())
        session['user_id'] = user_id

    try:
        if user_id not in user_chains:            
            vectorstore = vectorstore_builder.load_vectorstore('faiss_index_openai')
            llm = ChatOpenAI(temperature=0.7)
            chain = conversation_chain.get_chat_response(vectorstore, llm)
            user_chains[user_id] = chain
        user_chain = user_chains[user_id]
        result = user_chain.invoke({"question": user_question})
        answer = result.get('answer', 'I generated a response, but could not parse the format.')
        
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': f'An error occurred while generating the answer: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)