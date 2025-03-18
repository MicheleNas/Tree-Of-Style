import json 
from collections import Counter 
from nltk.util import ngrams 
from transformers import pipeline 
import os 
import matplotlib.pyplot as plt
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configura Hugging Face 
hf_token = os.getenv("HUGGING_FACE_TOKEN") 
bert_italian_pos = pipeline( "token-classification", model="sachaarbonel/bert-italian-cased-finetuned-pos", token=hf_token, device=device ) 


# Funzione per estrarre i POS da una frase 
def extract_pos_tags(sentence): 
    """ 
    Esegue il POS tagging sulla frase e restituisce una lista di tag POS. 
    """ 
    pos_results = bert_italian_pos(sentence) 
    return [result["entity"] for result in pos_results]  # Estrae solo i tag POS 

# Funzione per generare n-grammi e calcolare distribuzioni 
def analyze_pos_ngrams(data, attribute, n): 
    """ 
    Genera n-grammi basati sui tag POS delle frasi e calcola la distribuzione. 
    """ 
    all_ngrams = [] 
    for item in data: 
        pos_tags = extract_pos_tags(item[attribute])  # Estrai i POS 
        all_ngrams.extend(ngrams(pos_tags, n))  # Genera gli n-grammi 
    
    ngram_counts = Counter(all_ngrams)
    total_ngrams = sum(ngram_counts.values())  # Totale degli n-grammi
    
    # Calcola la distribuzione relativa
    ngram_distribution = {ngram: count / total_ngrams for ngram, count in ngram_counts.items()}
    
    return ngram_distribution

# Funzione per generare un grafico della distribuzione degli n-grammi
def plot_ngram_distribution(ngram_distribution, n):
    """Genera un grafico a barre per la distribuzione degli n-grammi."""
    # Ordina i n-grammi per distribuzione decrescente
    sorted_ngrams = sorted(ngram_distribution.items(), key=lambda item: item[1], reverse=True)
    
    # Prendi i primi 10 n-grammi
    top_ngrams = sorted_ngrams[:10]
    
    # Estrai i n-grammi e le relative distribuzioni
    ngram_labels = [' '.join(ngram) for ngram, _ in top_ngrams]
    distributions = [dist for _, dist in top_ngrams]
    
    # Crea il grafico a barre
    plt.figure(figsize=(10, 6))
    plt.barh(ngram_labels, distributions, color='skyblue')
    plt.xlabel('Distribuzione')
    plt.title(f'Distribuzione dei {n}-grammi POS')
    plt.gca().invert_yaxis()  # Inverti l'asse Y per avere il n-gramma più frequente in cima
    plt.show()



if __name__ == "__main__":
# Esegui lo script
    file_path = "./AlterEgo/data/dataset/sentences_all.json"  # Cambia con il percorso del tuo file
    
    # Caricamento delle frasi
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Analisi n-grammi POS
    for n in range(1, 6):  # Unigrammi, bigrammi, trigrammi, ecc.
        print(f"\nDistribuzione dei {n}-grammi POS:")
        pos_ngram_distribution = analyze_pos_ngrams(data, "sentence", n)
        
        # Mostra i 10 n-grammi con la distribuzione più alta
        sorted_ngrams = sorted(pos_ngram_distribution.items(), key=lambda item: item[1], reverse=True)
        for ngram, dist in sorted_ngrams[:10]:  # Mostra i 10 n-grammi più comuni
            print(f"{' '.join(ngram)}: {dist:.4f}")  # Mostra la distribuzione con 4 decimali
        
        # Crea il grafico della distribuzione
        plot_ngram_distribution(pos_ngram_distribution, n)
