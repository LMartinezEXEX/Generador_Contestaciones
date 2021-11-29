from tester_utils import *

def predictions(sentences, model_name):
    model = get_fine_tunned_model(model_name)

    trigger_embeddings = get_trigger_embeddings(model, 'Data/clus_all.tsv', 1000)

    for sent in sentences:
        result = predict(model, trigger_embeddings, sent)

        print_prediction(trigger_embeddings, result)

def main():
    sentences = ["La Sinopharm es una vacuna muy buena!",
                "Cual es la vacuna mas efectiva?",
                "Vacunas aprobadas por el Gobierno",
                "Donde vacunarse",
                "El aluminio refleja la luz",
                "Bueno, Alberto dijo Salud o Economía. La salud para ellos, la ruina económica para nosotros. No aclaró.",
                "Parakfe mabda majefs dmaif aionadiuvn sadf didi"]

    predictions(sentences, 'bert-base-nli-stsb-mean-tokens-recipes_top_10')

if __name__ == "__main__":
    main()