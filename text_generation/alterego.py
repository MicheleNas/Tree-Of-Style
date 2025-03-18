import os
import json
import numpy as np
import concurrent.futures
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
from text_definition_part.sentence_level.sentence_length import analyze_sentence_lengths  # Assumiamo che questo sia il percorso corretto

# Modello di stile per il confronto
model_style_embedding = SentenceTransformer('AnnaWegmann/Style-Embedding')

# Inizializzo OpenAI passandogli l'api_key
with open("./utils/secret.json") as f:
    secrets = json.load(f)
    key = secrets["api_key"]
    organization_id = secrets["organization"]
    project_id = secrets["project"]

os.environ["OPENAI_API_KEY"] = key
client = OpenAI(
    organization=organization_id,
    project=project_id
)


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


def get_response(messages: list):
    # Utilizzo l'API chat completions
    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages = messages,
        temperature = 1.0  # 0.0 - 2.0
    )
    return response

def chat_with_chatgpt4o_mini(conversation):

    response = get_response(messages=conversation)

    # Aggiungi la risposta dell'assistente alla conversazione
    alterego_message = response.choices[0].message.content

    return alterego_message

# Funzione per verificare se lo stile della frase è conforme
def verify_style_similarity(message, style_emb, threshold=0.70):
    sentence_embedding = model_style_embedding.encode(message)
    similarity = util.cos_sim(sentence_embedding, style_emb)
    is_similar = similarity >= threshold  # Verifica se supera la soglia

    # Aggiungi un dizionario con frase, risultato e similarità
    return {'sentence': message.strip(),
            'result': is_similar.item(),  # Converte il valore in True/False
            'similarity': similarity.item()  # Converte in numero
            }


def generate_and_verify(conversation, style_embeddings):
    # Genera la risposta usando la funzione chat_with_chatgpt4o_mini
    response = chat_with_chatgpt4o_mini(conversation)

    # Calcola la lunghezza della risposta (in numero di parole o caratteri, a seconda del caso)
    response_length = len(response.split())  # Supponiamo che la lunghezza venga calcolata in parole

    # Determina quale embedding utilizzare basato sulla lunghezza della risposta
    if response_length <= style_embeddings["s_T"]:
        embedding_to_use = style_embeddings["avg_s_emb"]
    elif response_length >= style_embeddings["l_T"]:
        embedding_to_use = style_embeddings["avg_l_emb"]
    else:
        embedding_to_use = style_embeddings["avg_m_emb"]

    # Verifica la similarità dello stile con l'embedding selezionato
    style_verification = verify_style_similarity(response, embedding_to_use)
    
    return style_verification



def tree_of_style(user_input, conversation, style_embeddings):

    conversation.append({"role": "user", "content": user_input})
    
    # Esegui 10 richieste in parallelo e raccogli i risultati
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        # Avvia 10 future per la generazione e verifica delle risposte
        futures = [executor.submit(generate_and_verify, conversation.copy(), style_embeddings) for _ in range(20)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    # Trova la risposta con la similarità massima
    best_response = max(results, key=lambda x: x['similarity'])
    conversation.append({"role": "assistant", "content": best_response['sentence']}) 

    return best_response, conversation