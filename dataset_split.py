import json

# Percorsi dei file JSON
path_all = './AlterEgo/data/dataset/sentences_all.json'
path_test = './AlterEgo/data/dataset/sentences_test.json'
path_train = './AlterEgo/data/dataset/sentences_train.json'

# Carica i dati dal file sentences_all.json
with open(path_all, 'r', encoding='utf-8') as file_all:
    frasi_all = json.load(file_all)

# Carica i dati dal file sentences_test.json
with open(path_test, 'r', encoding='utf-8') as file_test:
    frasi_test = json.load(file_test)

# Crea un set delle frasi nel file di test per un confronto più rapido
test_texts = {item['text'] for item in frasi_test}

# Filtra le frasi che non sono presenti nel file di test
train_data = [
    {
        "sentence": item["sentence"],
        "text": item["text"]
    }
    for item in frasi_all if item["text"] not in test_texts
]

# Scrivi il risultato nel file sentences_train.json
with open(path_train, 'w', encoding='utf-8') as file_train:
    json.dump(train_data, file_train, ensure_ascii=False, indent=4)

print(f"File {path_train} creato con successo.")
