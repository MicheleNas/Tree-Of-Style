import json
import os
from transformers import pipeline
import string
import spacy
import torch
from datasets import Dataset

# Carica il modello di lingua italiana
nlp = spacy.load("it_core_news_sm")

# Caricamento del token di Hugging Face
hf_token = os.getenv("HUGGING_FACE_TOKEN")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Inizializzazione del pipeline per il POS tagging
bert_italian_pos = pipeline("token-classification", model="sachaarbonel/bert-italian-cased-finetuned-pos", token=hf_token, device=device)

# Caricamento del vocabolario
with open('./data/nvdb/vocabolario_categorizzato.json', 'r') as f:
    vocabolario = json.load(f)

# Creazione di un dizionario delle parole categorizzate
categorie = {'fondamentale': {}, 'alto uso': {}, 'alta disponibilità': {}}
for entry in vocabolario:
    word = entry['word']
    tag = entry['tag']
    pos_list = set(entry['pos'])
    
    if tag == 'fondamentale':
        categorie['fondamentale'][word] = pos_list
    elif tag == 'alto uso':
        categorie['alto uso'][word] = pos_list
    elif tag == 'alta disponibilità':
        categorie['alta disponibilità'][word] = pos_list

# Funzione per cercare una parola nella categoria corretta
def search_category_word(word, pos, conteggio):
    flag_categorizzata = False
    for categoria, vocab in categorie.items():
        if word in vocab:
            if pos in vocab[word]:  # Controlliamo se il POS è valido per questa parola
                if categoria == 'fondamentale':
                    conteggio['fondamentale'] += 1
                elif categoria == 'alto uso' and conteggio['fondamentale'] == 0:
                    conteggio['alto uso'] += 1
                elif categoria == 'alta disponibilità' and conteggio['fondamentale'] == 0 and conteggio['alto uso'] == 0:
                    conteggio['alta disponibilità'] += 1
                flag_categorizzata = True
                break
    
    if not flag_categorizzata:
        if word in categorie['fondamentale']:
            conteggio['fondamentale'] += 1
            flag_categorizzata = True
        elif word in categorie['alto uso']:
            conteggio['alto uso'] += 1
            flag_categorizzata = True
        elif word in categorie['alta disponibilità']:
            conteggio['alta disponibilità'] += 1
            flag_categorizzata = True

    return conteggio, flag_categorizzata

# Funzione per analizzare una frase
def analizza_frase(frase, tokens):
    conteggio = {'fondamentale': 0, 'alto uso': 0, 'alta disponibilità': 0}
    totale_parole_frase = 0
    parole_non_categorizzate = []

    for token in tokens:
        word = token['word'].lower()
        pos = token['entity']

        if word in string.punctuation:
            continue

        totale_parole_frase += 1
        conteggio, flag_categorizzata = search_category_word(word, pos, conteggio)

        if not flag_categorizzata:
            word_lemmatizzata = nlp(word)[0].lemma_
            conteggio, flag_categorizzata = search_category_word(word_lemmatizzata, pos, conteggio)

            if not flag_categorizzata:
                parole_non_categorizzate.append(word_lemmatizzata)

    return conteggio, totale_parole_frase, parole_non_categorizzate

# Funzione principale per analizzare tutte le frasi
def conta_parole_categoria(frasi, attribute):
    conteggi_finali = {'fondamentale': 0, 'alto uso': 0, 'alta disponibilità': 0}
    totale_parole = 0
    tutte_parole_non_categorizzate = []

    # Creiamo un batch di frasi (Dataset Hugging Face)
    batch_frasi = [frase_dict[attribute] for frase_dict in frasi]

    # Creiamo un Dataset Hugging Face
    dataset = Dataset.from_dict({attribute: batch_frasi})

    # Elabora le frasi in batch usando il dataset
    results = bert_italian_pos(dataset[attribute])

    # Iteriamo sui risultati per ottenere conteggi, totale parole, e parole non categorizzate
    for frase, tokens in zip(batch_frasi, results):
        conteggio_frase, totale_parole_frase, parole_non_categorizzate = analizza_frase(frase, tokens)

        for categoria in conteggi_finali:
            conteggi_finali[categoria] += conteggio_frase[categoria]

        totale_parole += totale_parole_frase
        tutte_parole_non_categorizzate.extend(parole_non_categorizzate)

    percentuali_finali = {
        categoria: (conteggio / totale_parole) if totale_parole > 0 else 0
        for categoria, conteggio in conteggi_finali.items()
    }

    percentuale_non_categorizzate = (len(tutte_parole_non_categorizzate) / totale_parole) if totale_parole > 0 else 0
    percentuali_finali['non categorizzate'] = percentuale_non_categorizzate

    return percentuali_finali, totale_parole, tutte_parole_non_categorizzate

# Eseguiamo il conteggio e memorizziamo i risultati
if __name__ == "__main__":
    with open('./data/dataset/sentences_all.json', 'r') as f:
        frasi = json.load(f)

    conteggi, totale_parole, parole_non_categorizzate = conta_parole_categoria(frasi, attribute="sentence")

    # Mostriamo il risultato
    print(f"Conteggi delle categorie: {conteggi}")
    print(f"Totale delle parole: {totale_parole}")
    print(f"Numero di parole non categorizzate: {len(parole_non_categorizzate)}")

    with open('./data/nvdb/parole_non_categorizzate.json', 'w') as f:
        json.dump(parole_non_categorizzate, f, ensure_ascii=False, indent=4)
