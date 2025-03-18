import json
from text_generation.alterego import tree_of_style, split_transcripts_by_length
import zmq


#Prompt configurazione iniziale
with open('./utils/prompt.txt', 'r') as file:
    prompt = file.read().strip()

# Definisce il messaggio di sistema iniziale per la conversazione
initial_conversation = [
    {"role": "system", 
    "content": prompt
    }
]

# Creazione del contesto ZMQ
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

# Ottieni lo stile embedding dal path fornito
path_transcripts = "./data/dataset/sentences_all.json"
with open(path_transcripts, 'r') as file:
    data = json.load(file)

conversation = initial_conversation.copy()

# Restituisce embedding short, medium, long e le threshold short e long
style_embeddings = split_transcripts_by_length(data)

while True:
    # Aspetta la prossima richiesta dal client
    message = socket.recv()
    user_input = message.decode()
    print(f"Received request: {user_input}")

    # Controlla se l'input è "nuova conversazione"
    if user_input.lower() == "nuova conversazione":
        conversation = initial_conversation.copy()  # Reset della conversazione
        print("La conversazione è stata resettata.")
        
        # Invia una risposta di conferma al client
        socket.send_string("Conversazione resettata. Invia un nuovo messaggio.")
        continue  # Salta il resto del ciclo per attendere il prossimo messaggio

    
    if user_input.lower() == "quit":
        print("Chiusura della connessione e del contesto ZeroMQ...")
        socket.send_string("Connessione chiusa.")
        socket.close()  # Chiude il socket ZeroMQ
        context.term()   # Termina il contesto ZeroMQ
        break
        
    

    response, conversation = tree_of_style((user_input + ". Dai solo la risposta di Irene"), conversation, style_embeddings)

    print(f"Irene: {response['sentence']}") 
    print(f"Similarità: {response['similarity']:.4f}")

    # Invia la risposta al client
    socket.send_string(response['sentence'])  # Invia la risposta generata