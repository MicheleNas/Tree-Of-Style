import json
import spacy
from collections import Counter, OrderedDict

# Caricamento del modello in italiano di spaCy
nlp = spacy.load("it_core_news_sm")

# Funzione per la pre-elaborazione delle frasi
def preprocessa_frase(frase):
    # Creazione di un oggetto Doc tramite spaCy
    doc = nlp(frase)
    # Rimuove le stopwords e mantiene solo parole alfanumeriche, poi lemmatizza
    parole_pulite = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return parole_pulite



def count_lemmi(sentences, attribute):
    # Lista per raccogliere tutte le parole pulite
    parole_totali = []
    # Itera su tutti gli oggetti nel file JSON
    for item in sentences:
        # Usa solo il campo 'sentence'
        frase_completa = item[attribute]
        
        # Pre-elabora la frase
        parole = preprocessa_frase(frase_completa)
        parole_totali.extend(parole)

    # Conta la frequenza delle parole
    frequenze_lemmi = Counter(parole_totali)

    # Ordina la frequenza dei lemmi in ordine decrescente e crea un dizionario ordinato
    frequenze_lemmi_ordinate = OrderedDict(frequenze_lemmi.most_common())

    return frequenze_lemmi_ordinate




if __name__ == "__main__":
    # Caricamento del file JSON
    with open('./AlterEgo/data/dataset/sentences_all.json', 'r') as f:
        data = json.load(f)
    
    frequenze_lemmi = count_lemmi(data, "sentence")

    # Mostra le 10 parole più frequenti
    print(frequenze_lemmi)