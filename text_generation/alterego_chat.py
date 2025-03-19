import json
from text_generation.alterego import tree_of_style, split_transcripts_by_length
 
# Prompt configurazione iniziale
with open('./utils/prompt.txt', 'r') as file:
    prompt = file.read().strip()

conversation = [{"role": "system", "content": prompt }]

# Ottieni lo stile embedding dal path fornito 
path_transcripts = "./data/dataset/sentences_all.json"
with open(path_transcripts, 'r') as file:
    data = json.load(file)

# Restituisce embedding short, medium, long e le threshold short e long 
style_embeddings = split_transcripts_by_length(data) 

print("Sistema avviato. Digita 'nuova conversazione' per resettare o 'quit' per uscire.") 

while True: 
    user_input = input("Tu: ").strip()

    if user_input.lower() == "nuova conversazione": 
        conversation = [{"role": "system", "content": prompt }]  # Reset della conversazione 
        print("Conversazione resettata. Invia un nuovo messaggio.") 
        continue 

    if user_input.lower() == "quit": 
        print("Chiusura del sistema.") 
        break
        
    response, conversation = tree_of_style((user_input + ". Dai solo la risposta di Irene"), conversation, style_embeddings)

    print(f"Irene: {response['sentence']}") 
    print(f"Similarità: {response['similarity']:.4f}")