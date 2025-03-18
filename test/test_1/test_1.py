import os
import json
import numpy as np
import concurrent.futures
from openai import OpenAI
from groq import Groq
from sentence_transformers import SentenceTransformer, util
from text_definition_part.sentence_level.sentence_length import analyze_sentence_lengths  # Assumiamo che questo sia il percorso corretto
import time
from tqdm import tqdm

# Modello di stile per il confronto
model_style_embedding = SentenceTransformer('AnnaWegmann/Style-Embedding')

# Inizializzo OpenAI passandogli l'api_key
with open("./utils/secret.json") as f:
    secrets = json.load(f)
    key = secrets["api_key"]
    organization_id = secrets["organization"]
    project_id = secrets["project"]

    groq_key = secrets["api_key_groq"]

os.environ["OPENAI_API_KEY"] = key
client_openai = OpenAI(
    organization=organization_id,
    project=project_id
)


client_groq = Groq(
    api_key=groq_key,
)

# Funzione per ottenere una risposta da GPT
def get_gpt_response(messages: list):
    response = client_openai.chat.completions.create(
        model="gpt-4o-mini",  # Specifica il modello da utilizzare
        messages=messages,
        temperature=1.0  # 0.0 - 2.0
    )
    return response

# Funzione per ottenere una risposta da Groq
def get_groq_response(messages: list, model="llama-3.3-70b-versatile"):
    chat_completion = client_groq.chat.completions.create(
        messages=messages,
        model= model,  # Specifica il modello di Groq
    )
    return chat_completion

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

def split_transcripts_by_length(data, attribute="sentence"):
    # Analizza le frasi per ottenere media, varianza, deviazione standard e lunghezze
    media, _, deviazione_standard, _ = analyze_sentence_lengths(data)
    
    # Calcoliamo i gruppi in base alla lunghezza media delle frasi
    breve_threshold = media - deviazione_standard
    lunga_threshold = media + deviazione_standard
    
    short_sentences = []
    medium_sentences = []
    long_sentences = []

    for element in data:
        sentence = element[attribute]
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
            "s_T": breve_threshold, 
            "l_T": lunga_threshold}

def chat_with_chatgpt4o_mini(conversation):
    response = get_gpt_response(conversation)
    alterego_message = response.choices[0].message.content
    return alterego_message

def chat_with_groq(conversation, model):
    response = get_groq_response(conversation, model)
    # Modifica per ottenere correttamente la risposta
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

def generate_and_verify(conversation, style_embeddings, model=None):
    # Genera la risposta usando la funzione chat_with_chatgpt4o_mini o chat_with_groq
    if model in ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"]:
        response = chat_with_groq(conversation, model)
    else:
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

def alterego(user_input, conversation, style_embeddings, model):
    conversation.append({"role": "user", "content": user_input})
    
    # Esegui 5 richieste in parallelo e raccogli i risultati
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Avvia 5 future per la generazione e verifica delle risposte
        futures = [executor.submit(generate_and_verify, conversation.copy(), style_embeddings, model) for _ in range(5)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    # Trova la risposta con la similarità massima
    best_response = max(results, key=lambda x: x['similarity'])
    conversation.append({"role": "assistant", "content": best_response['sentence']}) 

    return best_response, conversation












path_test = './data/dataset/sentences_test.json'
path_train = './data/dataset/sentences_train.json'

# Prompt configurazione iniziale
with open('./utils/prompt.txt', 'r') as file:
    prompt = file.read().strip()

conversation = [{"role": "system", "content": prompt }]

# Ottieni lo stile embedding dal path fornito 
with open(path_train, 'r') as file:
    data_train = json.load(file)

# Restituisce embedding short, medium, long e le threshold short e long 
style_embeddings = split_transcripts_by_length(data_train) 


with open(path_test, 'r') as file:
    data_test = json.load(file)




# Lista dei modelli da usare
models = ["llama-3.3-70b-versatile", "gpt-4o-mini","mixtral-8x7b-32768"]

# Lista di dizionari per memorizzare i risultati modificati
results = []

# Itera attraverso ogni modello
for model in models:
    # Per ogni modello, itera attraverso i dati di test
    for item in tqdm(data_test, desc=f"Elaborazione {model}", unit="item"):
        conversation = [{"role": "system", "content": prompt}]
        user_input = item["text"].strip()
        status = item["status"]

        if status:
            response, _ = alterego(f"{user_input}. Status di Irene: {status}. Rispondi tenendo in considerazione lo status. Dai solo la risposta di Irene secondo il suo stile.", conversation, style_embeddings, model)
        else:
            response, _ = alterego(f"{user_input}. Dai solo la risposta di Irene secondo il suo stile.", conversation, style_embeddings, model)

        # Aggiungi le informazioni nella struttura dell'elemento
        item["Alterego_Sentence"] = response["sentence"]
        item["similarity"] = response["similarity"]

        # Aggiungi l'elemento modificato alla lista dei risultati
        results.append(item)

        if model in ["llama-3.3-70b-versatile","mixtral-8x7b-32768"]:
            # Pausa tra le richieste (se vuoi aggiungere un ritardo)
            time.sleep(60)

    # Salva il risultato per il modello corrente in un file JSON
    output_path = f'test/test_1/results/{model}_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Risultati salvati per {model} in {output_path}")
    
    # Resetta la lista dei risultati per il prossimo modello
    results = []