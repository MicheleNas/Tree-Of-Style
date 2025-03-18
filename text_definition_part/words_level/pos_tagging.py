
import os
from dotenv import load_dotenv

from collections import Counter
from transformers import pipeline

load_dotenv()

hf_token = os.getenv("HUGGING_FACE_TOKEN")

bert_italian_pos = pipeline("token-classification", model="sachaarbonel/bert-italian-cased-finetuned-pos", token=hf_token)


def calculate_pos(testo):
  # Trovo i tag
  pos_tags = bert_italian_pos(testo)

  # Estrarre le parole e i corrispondenti tag POS
  tag_pos = [element['entity'] for element in pos_tags]

  # Passo 3: Classificare e analizzare i tag POS
  count_tag = Counter(tag_pos)

  # Separare parole di contenuto e parole di funzione
  tag_content_words = {'ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN', 'VERB'}
  tag_function_words = {'ADP', 'AUX', 'CCONJ', 'DET', 'NUM', 'PART', 'PRON', 'SCONJ', 'SYM', 'X', 'PUNCT'}

  # Calcolare il numero totale di parole nel testo
  total_words = sum(count_tag.values())

  content_word_tags= {tag: count_tag[tag]/total_words  for tag in tag_content_words if tag in count_tag}
  function_word_tags= {tag: count_tag[tag]/total_words for tag in tag_function_words if tag in count_tag}

  return {'content_words' : content_word_tags, 'function_words' : function_word_tags}


def calculate_avg_pos(lista):
    # Inizializza dizionari per le somme delle frequenze
    somma_content_words = {}
    somma_function_words = {}

    # Numero totale di righe nella lista
    num_righe = len(lista)

    # Itera attraverso tutti gli elementi nella lista
    for i, elemento in enumerate(lista):
        # Somma totale dei valori per verifica
        somma_totale_content = sum(elemento['content_words'].values())
        somma_totale_function = sum(elemento['function_words'].values())
        somma_totale_riga = somma_totale_content + somma_totale_function

        # Processa le parole di contenuto
        for pos in set(somma_content_words) | set(elemento['content_words']):
            somma_content_words[pos] = somma_content_words.get(pos, 0) + elemento['content_words'].get(pos, 0)

        # Processa le parole funzionali
        for pos in set(somma_function_words) | set(elemento['function_words']):
            somma_function_words[pos] = somma_function_words.get(pos, 0) + elemento['function_words'].get(pos, 0)

    # Calcola le medie per le parole di contenuto
    medie_content_words = {pos: somma / num_righe for pos, somma in somma_content_words.items()}

    # Calcola le medie per le parole funzionali
    medie_function_words = {pos: somma / num_righe for pos, somma in somma_function_words.items()}

    return {'content_words': medie_content_words, 'function_words': medie_function_words}


def print_pos(valori):
    print("Dai risultati ottengo:")
    print("Content words:")
    for pos, valore in valori['content_words'].items():
        print(f"- {pos}: {valore:.4f}")
    
    print("\nFunction words:")
    for pos, valore in valori['function_words'].items():
        print(f"- {pos}: {valore:.4f}")