from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import nltk
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__, template_folder='.')


@app.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.get_json()
    question = data.get('question')
    answer = data.get('answer')

    # Inicializar o pipeline de pontuação de respostas do Hugging Face
    nlp = pipeline('question-answering')
    result = nlp(question=question, context=answer)

    # Extrair palavras-chave da pergunta
    stop_words = set(stopwords.words('portuguese'))
    question_words = nltk.word_tokenize(question)
    keywords = [word for word in question_words if not word in stop_words]

    # Verificar se as palavras-chave estão na resposta
    keyword_score = sum([1 for word in keywords if word in answer.lower()])

    # Calcular a pontuação com base no comprimento da resposta
    length_score = min(10, len(answer) // 50)

    # A pontuação final é uma combinação da pontuação de palavras-chave, pontuação de comprimento e a confiança do modelo
    final_score = (keyword_score + length_score + result['score']) / 3

    return jsonify(score=final_score)


@app.route('/')
def home():
    return render_template('templates/index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
