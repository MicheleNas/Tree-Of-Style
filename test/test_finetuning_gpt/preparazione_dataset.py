import json

# Percorsi dei file di input e output
input_file = './AlterEgo/data/dataset/sentences_train.json'
output_file = './AlterEgo/data/dataset/finetuning_gpt.jsonl'

# Messaggio del sistema da usare per ogni entrata
system_message = (
    "Impersona Irene, 20 anni. "
    "Rispetta le caratteristiche grammaticali descritte nella sezione 'Stile linguistico'. "
    "Fornisci le informazioni contenute nella sezione 'Base di Conoscenza' solo quando ti viene fatta una domanda a riguardo. "
    "Usa un linguaggio semplice. Mantieni un tono naturale: scrivi come se parlassi normalmente."
)

# Funzione per convertire i dati nel formato richiesto
def create_jsonl_entry(user_text, assistant_text):
    return {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text}
        ]
    }

# Leggere il file JSON di input
with open(input_file, 'r', encoding='utf-8') as infile:
    data = json.load(infile)

# Aprire il file JSONL in modalità scrittura
with open(output_file, 'w', encoding='utf-8') as outfile:
    for entry in data:
        # Creare l'oggetto JSONL per ogni riga
        user_text = entry.get('text', '')
        assistant_text = entry.get('sentence', '')
        
        # Creare l'oggetto JSONL e scriverlo nel file
        jsonl_entry = create_jsonl_entry(user_text, assistant_text)
        outfile.write(json.dumps(jsonl_entry, ensure_ascii=False) + '\n')

print(f"File JSONL creato correttamente in: {output_file}")
