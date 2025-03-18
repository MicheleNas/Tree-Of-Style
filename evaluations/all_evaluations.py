from text_definition_part.words_level.nvbd_frequency import conta_parole_categoria
from text_definition_part.words_level.frequency_words import count_lemmi
from text_definition_part.words_level.pos_frequency import analyze_pos_ngrams
from text_definition_part.sentence_level.sentence_length import analyze_sentence_lengths
from text_definition_part.sentence_level.sentiment_analysis import sentiment_analysis
import json
import numpy as np
from scipy.spatial.distance import jensenshannon
from sentence_transformers import SentenceTransformer, util

import torch

# Carica il modello pre-addestrato
model_st = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')


def js_similarity(D1, D2):
    return 1 - jensenshannon(D1, D2)

def sentences_evaluations(data, attribute):
    # Dizionario per raccogliere i risultati
    results = {}

# ----------------------------------------------------------------------- WORDS LEVEL -----------------------------------------------------------------------

# ----------------------------------------------------------------------- nvbd -----------------------------------------------------------------------

    # Eseguiamo il conteggio
    conteggi_0, _, _ = conta_parole_categoria(data, attribute[0])
    conteggi_1, _, _ = conta_parole_categoria(data, attribute[1])

    # Calcolare la similarità di Jensen-Shannon
    results["Sim-js_conteggi"] = js_similarity(np.array(list(conteggi_0.values())), np.array(list(conteggi_1.values())))

# ----------------------------------------------------------------------- frequency words -----------------------------------------------------------------------
    #Conteggio dei lemmi
    frequenze_lemmi_0 = count_lemmi(data, attribute[0])
    frequenze_lemmi_1 = count_lemmi(data, attribute[1])

    # Unire le chiavi (parole) da entrambe le distribuzioni
    tutte_le_parole = set(frequenze_lemmi_0.keys()).union(set(frequenze_lemmi_1.keys()))

    # Creare dizionari con 0 come valore per le parole mancanti
    conteggi_0 = {parola: frequenze_lemmi_0.get(parola, 0) for parola in tutte_le_parole}
    conteggi_1 = {parola: frequenze_lemmi_1.get(parola, 0) for parola in tutte_le_parole}

    # Convertire i dizionari in array di frequenze (assicurandosi che siano float64)
    distribuzione_0 = np.array(list(conteggi_0.values()), dtype=np.float64)
    distribuzione_1 = np.array(list(conteggi_1.values()), dtype=np.float64)

    # Normalizzare le distribuzioni (dividere per la somma totale di ciascuna distribuzione)
    somma_0 = np.sum(distribuzione_0)
    somma_1 = np.sum(distribuzione_1)

    # Evitare divisioni per zero nel caso in cui la somma sia zero
    if somma_0 > 0:
        distribuzione_0 /= somma_0
    if somma_1 > 0:
        distribuzione_1 /= somma_1

    # Aggiungiamo i risultati dei lemmi
    results["Sim-js_lemmi"] = js_similarity(distribuzione_0, distribuzione_1)

# ----------------------------------------------------------------------- pos frequency -----------------------------------------------------------------------

    for n in range(1, 6):  # Unigrammi, bigrammi, trigrammi, ecc.
        # Calcolare la distribuzione degli n-grammi POS
        pos_ngram_distribution_0 = analyze_pos_ngrams(data, attribute[0], n)
        pos_ngram_distribution_1 = analyze_pos_ngrams(data, attribute[1], n)
        
        # Unire le chiavi (n-grammi POS) da entrambe le distribuzioni
        tutte_le_pos_ngram = set(pos_ngram_distribution_0.keys()).union(set(pos_ngram_distribution_1.keys()))
        
        # Creare dizionari con 0 come valore per gli n-grammi mancanti
        conteggi_0 = {ngram: pos_ngram_distribution_0.get(ngram, 0) for ngram in tutte_le_pos_ngram}
        conteggi_1 = {ngram: pos_ngram_distribution_1.get(ngram, 0) for ngram in tutte_le_pos_ngram}

        # Convertire i dizionari in array di frequenze
        distribuzione_0 = np.array(list(conteggi_0.values()), dtype=np.float64)
        distribuzione_1 = np.array(list(conteggi_1.values()), dtype=np.float64)
        
        results[f"Sim-js_{n}-grammi_POS"] = js_similarity(distribuzione_0, distribuzione_1)



# ----------------------------------------------------------------------- SENTENCE LEVEL -----------------------------------------------------------------------

# ----------------------------------------------------------------------- sentence_length -----------------------------------------------------------------------

    # Calcolare la distribuzione delle lunghezze delle frasi
    _, _, _, distribuzione_lunghezze_0 = analyze_sentence_lengths(data, attribute[0])
    _, _, _, distribuzione_lunghezze_1 = analyze_sentence_lengths(data, attribute[1])

    # Calcolare la somma totale dei conteggi per ciascuna distribuzione
    somma_0 = np.sum(distribuzione_lunghezze_0)
    somma_1 = np.sum(distribuzione_lunghezze_1)

    # Normalizzare le distribuzioni
    # Evitare divisioni per zero nel caso in cui la somma sia zero
    if somma_0 > 0:
        distribuzione_lunghezze_0 = np.array(distribuzione_lunghezze_0, dtype=np.float64) / somma_0
    else:
        distribuzione_lunghezze_0 = np.array(distribuzione_lunghezze_0, dtype=np.float64)

    if somma_1 > 0:
        distribuzione_lunghezze_1 = np.array(distribuzione_lunghezze_1, dtype=np.float64) / somma_1
    else:
        distribuzione_lunghezze_1 = np.array(distribuzione_lunghezze_1, dtype=np.float64)

    # Aggiungere il risultato al dizionario principale
    results["Sim-js_lunghezze_frasi"] = js_similarity(distribuzione_lunghezze_0, distribuzione_lunghezze_1)

# ----------------------------------------------------------------------- sentiment_analysis -----------------------------------------------------------------------

    # Analisi del sentiment
    average_scores_0 = sentiment_analysis(data, attribute[0])
    average_scores_1 = sentiment_analysis(data, attribute[1])

    # Convertire le distribuzioni in array numpy per il calcolo della divergenza
    distribuzione_0_array = np.array(list(average_scores_0.values()), dtype=np.float64)
    distribuzione_1_array = np.array(list(average_scores_1.values()), dtype=np.float64)

    # Aggiungere il risultato al dizionario principale
    results["Sim-js_sentiment"] = js_similarity(distribuzione_0_array, distribuzione_1_array)

# ----------------------------------------------------------------------- SBERT -----------------------------------------------------------------------

# ----------------------------------------------------------------------- cos_sim content -----------------------------------------------------------------------

    # Crea le liste per le frasi
    alterego_sentences = []
    sentence = []

    # Itera sugli elementi del JSON per estrarre le frasi
    for item in data:
        alterego_sentences.append(item[attribute[1]])
        sentence.append(item[attribute[0]])

    # Calcola gli embeddings per entrambe le liste
    Alterego_embeddings = torch.tensor(model_st.encode(alterego_sentences))
    sentence_embeddings = torch.tensor(model_st.encode(sentence))

    # Calcola la similarità coseno per ogni coppia
    cos_sim = util.cos_sim(Alterego_embeddings, sentence_embeddings)

    # Calcola la media della diagonale principale di cos_sim
    diagonal_values = cos_sim.diagonal()  # Estrai i valori sulla diagonale principale
    mean_cos_sim = diagonal_values.mean().item()  # Calcola la media e convertila in float

    results["Avg-Sim-cos_Content"] = mean_cos_sim

    # Libera la memoria GPU non utilizzata
    torch.cuda.empty_cache()

    return results



#if __name__=="__main__":
#    # Caricamento dei file JSON
#    with open('./AlterEgo/test/test_2/results/gpt-4o-mini_ToS_results.json', 'r') as f:
#        data = json.load(f)
#
#    results = sentences_evaluations(data, ["sentence", "Alterego_Sentence"])
#
#    # Stampare i risultati finali
#    print(results)


import os
import csv

# Funzione per ottenere il modello e la tecnica dal path del file
def extract_model_and_technique(file_path):
    # Supponiamo che il nome del file contenga informazioni sul modello e sulla tecnica
    # Es: "gpt-4o-mini_ToS_results.json" o "gpt-4o-mini_Zero-shot_results.json"
    file_name = os.path.basename(file_path)
    
    if "ToS" in file_name:
        technique = "ToS"
    elif "Zero-shot" in file_name:
        technique = "Zero-Shot"
    else:
        technique = "Unknown"  # Nel caso non ci fosse un'indicazione chiara
    
    # Prendiamo il modello dal nome del file, escludendo "_ToS_results.json" o "_Zero-shot_results.json"
    model = file_name.split('_')[0]
    
    return model, technique





# Funzione principale
if __name__ == "__main__":
    # Lista di path dei file JSON
    file_paths = [
        './AlterEgo/test/finetuning_gpt/results/ft:gpt-4o-mini-2024-07-18:alterego:AfrDa3fM_Zero-Shot_results.json',
        './AlterEgo/test/test_2/results/gpt-4o-mini_Zero-Shot_results.json',
        './AlterEgo/test/test_2/results/llama-3.3-70b-versatile_ToS_results.json',
        './AlterEgo/test/test_2/results/llama-3.3-70b-versatile_Zero-Shot_results.json'
    ]
    
    # Creazione del file CSV
    with open('./AlterEgo/test/finetuning_gpt/results/test.csv', mode='w', newline='') as csvfile:
        fieldnames = [
            'model', 'technique', 'Sim-js_conteggi', 'Sim-js_lemmi', 
            'Sim-js_1-grammi_POS', 'Sim-js_2-grammi_POS', 'Sim-js_3-grammi_POS', 
            'Sim-js_4-grammi_POS', 'Sim-js_5-grammi_POS', 'Sim-js_lunghezze_frasi', 
            'Sim-js_sentiment', 'Avg-Sim-cos_Content'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        i = 0

        # Iterazione sui file JSON
        for file_path in file_paths:
            print(i)
            # Caricamento dei dati JSON
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Calcolare i risultati usando sentences_evaluations
            results = sentences_evaluations(data, ["sentence", "Alterego_Sentence"])
            
            # Estrazione del modello e della tecnica dal path del file
            model, technique = extract_model_and_technique(file_path)
            
            # Scrivere i risultati nel CSV
            row = {
                'model': model,
                'technique': technique,
                'Sim-js_conteggi': results.get('Sim-js_conteggi', ''),
                'Sim-js_lemmi': results.get('Sim-js_lemmi', ''),
                'Sim-js_1-grammi_POS': results.get('Sim-js_1-grammi_POS', ''),
                'Sim-js_2-grammi_POS': results.get('Sim-js_2-grammi_POS', ''),
                'Sim-js_3-grammi_POS': results.get('Sim-js_3-grammi_POS', ''),
                'Sim-js_4-grammi_POS': results.get('Sim-js_4-grammi_POS', ''),
                'Sim-js_5-grammi_POS': results.get('Sim-js_5-grammi_POS', ''),
                'Sim-js_lunghezze_frasi': results.get('Sim-js_lunghezze_frasi', ''),
                'Sim-js_sentiment': results.get('Sim-js_sentiment', ''),
                'Avg-Sim-cos_Content': results.get('Avg-Sim-cos_Content', '')
            }
            writer.writerow(row)

        