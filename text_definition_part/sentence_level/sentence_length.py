import json
import numpy as np
import matplotlib.pyplot as plt

# Funzione che analizza le frasi da un dato JSON e restituisce le statistiche
def analyze_sentence_lengths(data, attribute="sentence"):
    # Estrai le frasi e calcola la lunghezza di ciascuna frase
    frasi = [item[attribute] for item in data if attribute in item]
    lunghezze = [len(frase.split()) for frase in frasi]
    # Calcola la media, la varianza e la deviazione standard delle lunghezze
    media = np.mean(lunghezze)
    varianza = np.var(lunghezze)
    deviazione_standard = np.std(lunghezze)
    
    return media, varianza, deviazione_standard, lunghezze

# Funzione per generare e mostrare il grafico
def crea_plot(lunghezze, media, deviazione_standard):
    plt.figure(figsize=(10, 6))
    plt.hist(lunghezze, bins=20, edgecolor='black', alpha=0.7)
    plt.axvline(media, color='red', linestyle='dashed', linewidth=2, label=f'Media: {media:.2f}')
    plt.axvline(media - deviazione_standard, color='blue', linestyle='dashed', linewidth=2, label=f'-1 STD: {media - deviazione_standard:.2f}')
    plt.axvline(media + deviazione_standard, color='blue', linestyle='dashed', linewidth=2, label=f'+1 STD: {media + deviazione_standard:.2f}')

    # Etichette e titolo del grafico
    plt.title('Distribuzione della Lunghezza delle Frasi')
    plt.xlabel('Numero di parole per frase')
    plt.ylabel('Frequenza')
    plt.legend()

    # Mostra il grafico
    plt.show()

# Blocco principale che viene eseguito se il file viene eseguito come programma principale
if __name__ == "__main__":
    path_json = './AlterEgo/data/dataset/frasi_telegram.json'

    # Carica i dati dal file JSON
    with open(path_json, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Calcola media, varianza, deviazione standard e ottieni le lunghezze
    media, varianza, deviazione_standard, lunghezze = analyze_sentence_lengths(data, "sentence")

    # Stampa i risultati
    print(f"Lunghezza media delle frasi: {media}")
    print(f"Varianza: {varianza}")
    print(f"Deviazione standard: {deviazione_standard}")

    # Crea e mostra il grafico
    crea_plot(lunghezze, media, deviazione_standard)