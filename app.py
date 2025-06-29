from flask import Flask, request, jsonify, render_template
from transformers import pipeline
from googletrans import Translator

app = Flask(__name__)
translator = Translator()

# Initialize BERT QA pipeline
qa_pipeline = pipeline("question-answering", model="bert-squad")

# Store context and chat history
chat_history = []
context_data = ""
model_name = "bert-squad"  # Default model

# Available models
available_models = {
    "BERT": "bert-squad",
    "Distil-BERT": "distilbert-squad",
    "ALBERT": "albert-squad",
    "INDOBERT": "indobert-squad",
}

@app.route('/')
def index():
    return render_template('index.html')

# Route to set context
@app.route('/set_context', methods=['POST'])
def set_context():
    global context_data, chat_history
    data = request.json
    context_data = data.get('context', '')
    chat_history = []  # Reset chat history when context is updated
    return jsonify({"message": "Context updated successfully."})

# Route to set the model
@app.route('/set_model', methods=['POST'])
def set_model():
    global qa_pipeline, model_name
    data = request.json
    selected_model = data.get('model', '')
    
    if selected_model in available_models:
        model_name = selected_model
        qa_pipeline = pipeline("question-answering", model=available_models[selected_model])
        return jsonify({"message": f"Model changed to {model_name}"})
    else:
        return jsonify({"message": "Model not found."}), 400

# Route to handle questions
@app.route('/ask_question', methods=['POST'])
def ask_question():
    global context_data, chat_history
    data = request.json
    question = data.get('question', '')
    lang = data.get('language', 'id')

    # Translate to English if not already
    if lang != 'en':
        translated = translator.translate(question, src=lang, dest='en')
        translated_question = translated.text
    else:
        translated_question = question


    # Prepare input by combining context and question
    qa_input = {
        'question': translated_question,
        'context': context_data
    }
    
    # Get the answer from the BERT model
    answer = qa_pipeline(qa_input)['answer']
    
    # Update chat history
    chat_history.append({
        'question': question,
        'answer': answer
    })

    print(answer)

    if lang != 'en':
        answer = translator.translate(answer, src='en', dest=lang).text
    
    return jsonify({
        'question': question,
        'answer': answer,
        'chat_history': chat_history
    })

if __name__ == '__main__':
    app.run(debug=True)
