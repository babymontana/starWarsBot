import re
import tensorflow as tf
from procesadorTexto import  ProcesadorTexto
import numpy as np


def oracion_a_secuencia(sentence, vocab_to_int):
    sentence = limpia(sentence)
    return [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in sentence.split()]

def limpia(texto):
        texto = texto.lower()
        texto = re.sub(r"\n", "", texto)
        texto = re.sub(r"[-()]", "", texto)
        texto = re.sub(r"\.", ".", texto)
        texto = re.sub(r"\!", "!", texto)
        texto = re.sub(r"\?", "?", texto)
        texto = re.sub(r"\,", ",", texto)
        texto = re.sub(r"i'm", "i am", texto)
        texto = re.sub(r"he's", "he is", texto)
        texto = re.sub(r"she's", "she is", texto)
        texto = re.sub(r"it's", "it is", texto)
        texto = re.sub(r"that's", "that is", texto)
        texto = re.sub(r"what's", "that is", texto)
        texto = re.sub(r"\'ll", " will", texto)
        texto = re.sub(r"\'re", " are", texto)
        texto = re.sub(r"won't", "will not", texto)
        texto = re.sub(r"can't", "cannot", texto)
        texto = re.sub(r"n'", "ng", texto)
        texto = re.sub(r"ahh", "ah", texto)
        texto = re.sub(r"i've", "i have", texto)
        texto = re.sub(r"there's", "there is", texto)
        texto = re.sub(r"you've", "you have", texto)
        texto = re.sub(r"there's", "there is", texto)

        return texto

def main():
    proces = ProcesadorTexto()

    input_sentence = "Captain Solo, do you copy?"

    # Formato al input
    input_sentence = oracion_a_secuencia(input_sentence, proces.vocabulario_fuente_int)
    checkpoint = "modelo.ckpt"
    checkpoint = "./" + checkpoint

    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        # Carga del grafo modelo
        loader = tf.train.import_meta_graph(checkpoint + '.meta')
        loader.restore(sess, checkpoint)
        # Carga de los tensores
        input_data = loaded_graph.get_tensor_by_name('input:0')
        logits = loaded_graph.get_tensor_by_name('logits:0')
        keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        # Ejecución del modelo
        response_logits = sess.run(logits, {input_data: [input_sentence], keep_prob: 1.0})[0]

    # Entrada del usuario
    print('Entrada')
    print('  Palabras entrada: {}'.format([proces.vfuente_int_diccionario[i] for i in input_sentence]))

    # Salida del Bot
    print('\nRespuesta')
    print('  Palabras del Bot: {}'.format([proces.vobjetivo_int_diccionario[i] for i in np.argmax(response_logits, 1)]))

if __name__ == '__main__':
    main()