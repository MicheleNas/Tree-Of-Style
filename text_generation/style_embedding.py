import numpy as np
from sentence_transformers import SentenceTransformer
from text_definition_part.sentence_level.sentence_length import analyze_sentence_lengths  # Assumiamo che questo sia il percorso corretto

def avg_embedding(sentences):
    model = SentenceTransformer('AnnaWegmann/Style-Embedding')
    
    # Genero gli embedding
    embeddings = model.encode(sentences)
    
    # Calcola l'embedding medio
    if embeddings.size > 0:
        average_embedding = np.mean(embeddings, axis=0)
    else:
        average_embedding = None
    
    return average_embedding



def split_transcripts_by_length(data):
    # Analizza le frasi per ottenere media, varianza, deviazione standard e lunghezze
    media, _, deviazione_standard, _ = analyze_sentence_lengths(data)
    
    # Calcoliamo i gruppi in base alla lunghezza media delle frasi
    breve_threshold = media - deviazione_standard
    lunga_threshold = media + deviazione_standard
    
    short_sentences = []
    medium_sentences = []
    long_sentences = []

    for element in data:
        sentence = element['sentence']
        length = len(sentence.split())  # Considera la lunghezza in parole (puoi cambiare questa metrica)

        if length <= breve_threshold:
            short_sentences.append(sentence)
        elif length >= lunga_threshold:
            long_sentences.append(sentence)
        else:
            medium_sentences.append(sentence)

    avg_short_embedding = avg_embedding(short_sentences)
    avg_medium_embedding = avg_embedding(medium_sentences)
    avg_long_embedding = avg_embedding(long_sentences)
    
    # Restituisci i risultati
    return {"avg_s_emb": avg_short_embedding, 
            "avg_m_emb": avg_medium_embedding, 
            "avg_l_emb": avg_long_embedding, 
            "s_T":breve_threshold, 
            "l_T": lunga_threshold}
