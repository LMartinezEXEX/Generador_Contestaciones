from tester_utils import *

def predictions(sentences, model_name):
    model = get_fine_tunned_model(model_name)

    trigger_embeddings = get_trigger_embeddings(model, 'Data/clus_all.tsv', 1000)

    #trigger_ans = get_trigger_answers("../Proyecto/Dataset/processed.csv")

    for sent in sentences:
        result = predict(model, trigger_embeddings, sent)

        answer = get_answer_prediction(trigger_embeddings, result)

        #eval = evaluate(sent, answer, trigger_ans)
        #print("EVAL:", eval)

def main():
    sentences = ["La Sinopharm es una vacuna muy buena!",
                "Cual es la vacuna mas efectiva?",
                "Vacunas aprobadas por el Gobierno",
                "Donde vacunarse",
                "El aluminio refleja la luz",
                "Bueno, Alberto dijo Salud o Economía. La salud para ellos, la ruina económica para nosotros. No aclaró.",
                "Parakfe mabda majefs dmaif aionadiuvn sadf didi"]
    
    #sentences = ["Monitor Público de Vacunación 15-03 2.488.218 dosis aplicadas Con una dosis 2.022.489 personas (4,46%) Con ambas dosis 465.729 personas (1,03%) https://t.co/UdXXPTrG3x Dosis aplicadas y distribuidas https://t.co/aQ2utO8man",
    #            "La ciencia es la única salida.   Aguanten las vacunas",
    #            "Ni loco me la pongo, andá a saber que le ponen https://t.co/trhvKTreVs",
    #            "Xq le siguen preguntando a @RubinsteinOk algo?",
    #            "Me estoy haciendo un poco hincha de River.",
    #            "Descubrieron un árbol petrificado en Lesbos de entre 17 y 20 millones de años de antigüedad 👇 https://t.co/e7LXOTnu0D"]

    predictions(sentences, 'bert-base-nli-stsb-mean-tokens-recipes')

if __name__ == "__main__":
    main()