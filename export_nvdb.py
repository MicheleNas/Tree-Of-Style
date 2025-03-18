import json
import pdfplumber
import re
import time

# Percorso del file PDF
pdf_path = "./AlterEgo/utils/nuovovocabolariodibase.pdf"
pos_json_path = "./AlterEgo/data/nvdb/nvdb_pos.json"

# Carica il file JSON con i POS
with open(pos_json_path, "r", encoding="utf-8") as f:
    pos_list = json.load(f)

# Estrai il dizionario dal primo elemento della lista
pos_dict = pos_list[0] if pos_list else {}

# Inizializza lista per le parole
words = []

# Funzione per classificare le parole in base allo stile
def classify_word(font_name):
    if "Bold" in font_name:  # Font in neretto
        return "fondamentale"
    elif "Italic" in font_name:  # Font in corsivo
        return "alta disponibilità"
    else:  # Font normale
        return "alto uso"

# Funzione per filtrare parole valide
def is_valid_word(word):
    # Regex per parole da escludere: "bloc. di comando", parole con punti finali, e "e"
    invalid_patterns = r"bloc\.\sdi\scomando\b|([a-z]+\.)+|^e$"
    
    # Restituisce True solo se la parola non corrisponde ai pattern da escludere
    return not re.search(invalid_patterns, word.lower())

# Variabile per memorizzare l'ultima parola spezzata
previous_word = None

# Apertura del file PDF
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        # Estrai il testo con le proprietà del font
        for word_data in page.extract_words(extra_attrs=["fontname"]):
            word = word_data["text"]
            font_name = word_data["fontname"]
            height = word_data["height"]
            
            # Considera solo le parole con height tra 10 e 10.50
            if not (10 <= height <= 10.50):
                continue  # Salta la parola se non è nell'intervallo

            # Rimuove i numeri accanto alle parole
            word = re.sub(r"\d+", "", word)

            # Gestisce le parole spezzate con trattino
            if previous_word and previous_word.endswith("-"):
                # Rimuove il trattino e unisce le parole
                word = previous_word[:-1] + word
                previous_word = None  # Reset
            elif word.endswith("-"):
                # Salva la parola spezzata per il prossimo ciclo
                previous_word = word
                continue
            else:
                previous_word = None

            if word.endswith(','):
                 word = word[:-1]

            # Filtra le parole valide (escludendo "e" e lettere singole maiuscole)
            if is_valid_word(word):
                # Classifica la parola e aggiungi POS tags
                word_info = {
                    "word": word,
                    "tag": classify_word(font_name),
                    "pos": []
                }
                words.append(word_info)
            else:
                # Se la parola non è valida, cerca i POS associati a questa parola
                for key, value in pos_dict.items():
                    if isinstance(key, str) and key in word:
                        # Aggiungi il POS all'ultimo elemento della lista words
                        if words:
                            # L'ultimo elemento inserito nella lista "words"
                            last_word_info = words[-1]
                            # Se il campo "pos" esiste, aggiungi il nuovo POS; altrimenti, inizializza una lista
                            if last_word_info["pos"] is None:
                                last_word_info["pos"] = []
                            last_word_info["pos"].append(value)
                        break


# Esporta in JSON
output_path = "./AlterEgo/data/nvdb/vocabolario_categorizzato.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(words, f, ensure_ascii=False, indent=2)

print(f"File JSON esportato in: {output_path}")
print("Dimensione lista parole: ", len(words))
