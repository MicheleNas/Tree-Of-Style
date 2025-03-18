from text_definition_part.words_level.nvbd_frequency import conta_parole_categoria
from text_definition_part.words_level.frequency_words import count_lemmi
from text_definition_part.words_level.pos_frequency import analyze_pos_ngrams
from text_definition_part.sentence_level.sentence_length import analyze_sentence_lengths
from text_definition_part.sentence_level.sentiment_analysis import sentiment_analysis
import json


def main(data, attribute):

    # ----------------------------------------------------------------------- WORDS LEVEL -----------------------------------------------------------------------

    # ----------------------------------------------------------------------- nvbd -----------------------------------------------------------------------
    # Dizionario per raccogliere i risultati
    results = {}

    # Eseguiamo il conteggio
    conteggi, totale_parole, parole_non_categorizzate = conta_parole_categoria(data, attribute)

    # Aggiungiamo i risultati nel dizionario
    results["nvbd"] = {
        "conteggi_categorie": conteggi,
        "totale_parole": totale_parole,
        "parole_non_categorizzate": len(parole_non_categorizzate),
    }


    # ----------------------------------------------------------------------- frequency words -----------------------------------------------------------------------

    # Conteggio dei lemmi
    frequenze_lemmi = count_lemmi(data, attribute)

    # Aggiungiamo i risultati dei lemmi
    results["frequenze_lemmi"] = frequenze_lemmi


    # ----------------------------------------------------------------------- pos frequency -----------------------------------------------------------------------

    # Analisi n-grammi POS
    pos_ngram_results = {}
    for n in range(1, 6):  # Unigrammi, bigrammi, trigrammi, ecc.
        pos_ngram_distribution = analyze_pos_ngrams(data, attribute, n)
        
        # Ordina gli n-grammi per frequenza
        sorted_ngrams = sorted(pos_ngram_distribution.items(), key=lambda item: item[1], reverse=True)
        
        # Aggiungiamo i primi 10 n-grammi al dizionario
        pos_ngram_results[f"{n}-grammi"] = [
            {"ngram": ' '.join(ngram), "distribuzione": dist} 
            for ngram, dist in sorted_ngrams[:10]
        ]

    results["pos_ngrammi"] = pos_ngram_results

    # ----------------------------------------------------------------------- SENTENCE LEVEL -----------------------------------------------------------------------

    # ----------------------------------------------------------------------- sentence_length -----------------------------------------------------------------------

    # Calcola media, varianza, deviazione standard e ottieni le lunghezze
    media, varianza, deviazione_standard, _ = analyze_sentence_lengths(data, attribute)

    # Aggiungiamo i risultati della lunghezza delle frasi
    results["lunghezza_frasi"] = {
        "media": media,
        "varianza": varianza,
        "deviazione_standard": deviazione_standard,
    }


    # ----------------------------------------------------------------------- sentiment_analysis -----------------------------------------------------------------------

    # Analisi del sentiment
    average_scores = sentiment_analysis(data, attribute)

    # Aggiungiamo i punteggi medi delle emozioni
    results["sentiment_analysis"] = average_scores

    # Scriviamo i risultati in un file JSON
    output_file = f'./evaluations/results/{attribute}_results_style.json'
    with open(output_file, 'w') as outfile:
        json.dump(results, outfile, indent=4)

    # Mostra i risultati in formato leggibile
    print(f"Risultati analizzati e salvati in './evaluations/results/{attribute}_results_style.json'")


if __name__=="__main__":
    # Caricamento dei file JSON
    with open('./data/results.json', 'r') as f:
        data = json.load(f)

    main(data, "sentence")