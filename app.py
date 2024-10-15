from flask import Flask, render_template, request, jsonify
import pickle
import re
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

def remove_mentions(text):
    """
    Supprime tous les mots commençant par un @ dans un texte.
    """
    return ' '.join(word for word in text.split() if not word.startswith('@'))

def clean_text_fct(text):
    """
    Nettoie le texte en supprimant les liens et les caractères spéciaux, sauf les smileys.
    """
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    pattern = re.compile(r'([^\w\s]|_)+', re.UNICODE)
    smileys = r'(:\)|:\(|:D|:P|:\*|;\)|;D|:\'-\(|<3|:\^])'
    text = re.sub(pattern, lambda x: x.group(0) if re.match(smileys, x.group(0)) else ' ', text)
    return text.strip()

def tokenizer_fct(sentence):
    """
    Tokenize une phrase en mots, en supprimant certains caractères spéciaux.
    """
    sentence_clean = sentence.replace('-', ' ').replace('+', ' ').replace('/', ' ').replace('#', ' ')
    #word_tokens = word_tokenize(sentence_clean)
    word_tokens = re.findall(r'\b\w+\b', sentence_clean)
    return word_tokens

def lower_start_fct(list_words):
    """
    Convertit tous les mots d'une liste en minuscules.
    """
    return [w.lower() for w in list_words]

def transform_text_fct(text, clean=True, tokenizer=True, lower=True):
    """
    Prépare un texte pour une représentation en sac de mots (bag of words) ou TF-IDF.
    """
    if clean:
        text = clean_text_fct(text)

    if tokenizer:
        text = tokenizer_fct(text)

    if lower:
        text = lower_start_fct(text)

    return ' '.join(text)

def get_embedding(text, embedding_model):
    """
    Retourne l'embedding correspondant pour le texte avec Tf-idf.
    """
    embedding = embedding_model.transform([text])
    scaler = StandardScaler(with_mean=False)
    return scaler.fit_transform(embedding)

# Dictionnaire temporaire pour stocker la prédiction associée à un tweet
prediction_cache = {}

# Charger le modèle Tf-idf et le modèle de classification MLPClassifier
with open('embeddings/tfidf_model.pkl', 'rb') as f:
    embedding_model = pickle.load(f)

with open('models/Tf-idf_MLP-Classifier_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET'])
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Récupérer les données JSON
    data = request.get_json()
    tweet_text = data.get('tweet_to_predict', '')

    # Nettoyage et prétraitement du texte
    cleaned_text = transform_text_fct(tweet_text)

    # Obtenir l'embedding correspondant (Tf-idf)
    embedding = get_embedding(cleaned_text, embedding_model)

    # Prédiction avec le modèle MLPClassifier
    probabilities = model.predict_proba(embedding)
    prediction = model.predict(embedding)

    result = "Positif" if prediction[0] == 1 else "Négatif"

    # Stocker la prédiction dans le cache pour vérification future
    prediction_cache[tweet_text] = result  # Stocker la prédiction pour ce tweet

    app.logger.debug('This is a debug log message - predict')
    app.logger.info('This is an information log message - predict')
    app.logger.warn('This is a warning log message - predict')
    app.logger.error('This is an error message - predict')
    app.logger.critical('This is a critical message - predict')

    return jsonify({'prediction': result})

@app.route('/feedback', methods=['POST'])
def feedback():
    # Récupérer le retour d'information de l'utilisateur
    data = request.get_json()
    tweet_text = data.get('tweet_to_predict')
    user_feedback = data.get('feedback')  # 'positive' ou 'negative'

    # Vérifier si la prédiction pour ce tweet est dans le cache
    predicted_label = prediction_cache.get(tweet_text)

    if predicted_label is not None:
        # Interprétation du feedback utilisateur
        if user_feedback == "positive":
            print(f"Correct prediction")
        elif user_feedback == "negative":
            print(f"Incorrect prediction")
    else:
        print(f'Erreur: aucune prédiction trouvée pour le tweet="{tweet_text}"')

    app.logger.debug('This is a debug log message - feedback')
    app.logger.info('This is an information log message - feedback')
    app.logger.warn('This is a warning log message - feedback')
    app.logger.error('This is an error message - feedback')
    app.logger.critical('This is a critical message - feedback')

    return jsonify({'status': 'success', 'message': 'Feedback enregistré'})

if __name__ == '__main__':
    app.run()
