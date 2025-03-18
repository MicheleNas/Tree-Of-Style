from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import json

# Caricare il modello e il tokenizer
tokenizer = AutoTokenizer.from_pretrained("aiknowyou/it-emotion-analyzer")
model = AutoModelForSequenceClassification.from_pretrained("aiknowyou/it-emotion-analyzer")

# Mappatura delle etichette personalizzate
label_map = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}

# Funzione per analizzare le emozioni con il modello
def analyze_emotions(text):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Tokenizzare il testo
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Sposta gli input sul dispositivo corretto
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Inferenza del modello
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Calcolare le probabilità con softmax
    probs = F.softmax(outputs.logits, dim=-1)
    
    # Creare i risultati con etichetta personalizzata e probabilità
    results = {label_map[i]: float(probs[0][i]) for i in range(len(label_map))}
    return results


def sentiment_analysis(data, attribute):
    
    # Inizializza variabili per calcolare la media degli score
    total_sentiments = {
        'joy': 0,
        'sadness': 0,
        'fear': 0,
        'anger': 0,
        'love': 0,
        'surprise': 0
    }
    
    # Conta il numero totale di frasi
    total_sentences = len(data)
    
    # Esegui l'analisi del sentiment per ogni frase
    for sentence_data in data:
        # Estrai la frase dall'oggetto JSON
        sentence = sentence_data[attribute]
        
        # Analizza la frase usando la funzione 'analyze_emotions'
        sentiment_scores = analyze_emotions(sentence)
        
        # Aggiungi i punteggi delle emozioni al totale
        for emotion in total_sentiments:
            total_sentiments[emotion] += sentiment_scores.get(emotion, 0)
    
    # Calcola la media per ogni emozione
    avg_sentiments = {emotion: score / total_sentences for emotion, score in total_sentiments.items()}
    
    return avg_sentiments


if __name__ == "__main__":

    file_path = "./data/dataset/sentences_all.json"

    # Carica il file JSON
    with open(file_path, 'r') as file:
        data = json.load(file)

    average_scores = sentiment_analysis(data, "sentence")

    # Stampa i punteggi medi
    print("Punteggi medi delle emozioni:")
    for emotion, score in average_scores.items():
        print(f"{emotion}: {score:.2f}")