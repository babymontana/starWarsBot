import re


class ProcesadorTexto:

    def __init__(self):
        texto = self.readTxt()
        texto_corto = self.textoCorto(texto)
        vocabulario = self.crearVocabulario(texto_corto)

        self.lm = 3
        contador = 0
        for k, v in vocabulario.items():
            if v >= self.lm:
                contador += 1

        self.vocabulario_fuente_int = self.vocabularioFuente(vocabulario)
        self.vocabulario_objetivo_int = self.vocabularioObjetivo(vocabulario)

        codigos_unicos = ['<PAD>', '<EOS>', '<UNK>', '<GO>']

        for c in codigos_unicos:
            self.vocabulario_fuente_int[c] = len(self.vocabulario_fuente_int) + 1
            self.vocabulario_objetivo_int[c] = len(self.vocabulario_objetivo_int) + 1



        self.vfuente_int_diccionario = {indice: v for v, indice in self.vocabulario_fuente_int.items()}
        self.vobjetivo_int_diccionario = {indice: v for v, indice in self.vocabulario_objetivo_int.items()}

        texto_fuente = texto_corto[:-1]
        texto_objetivo = texto_corto[1:]

        for i in range(len(texto_objetivo)):
            texto_objetivo[i] += ' <EOS>'
        texto_objetivo

        self.fuente_int = self.palabrasEnteros(texto_fuente, self.vocabulario_fuente_int)
        self.objetivo_int = self.objetivoInt(texto_objetivo, self.vocabulario_objetivo_int)

    def readTxt(self):
        texto = []
        file = open("starwars.txt", "r", encoding="UTF-8")
        for line in file:
            texto.append(self.limpia(line))
        return texto

    def textoCorto(self,texto):
        linea_extension = 50

        texto_corto = []
        for l in texto:
            if len(l.split()) <= linea_extension:
                texto_corto.append(l)
        return texto_corto

    def vocabularioFuente(self,vocabulario):

        # Vocabulario fuente a entero
        vocabulario_fuente_int = {}

        suma = 0
        for k, v in vocabulario.items():
            if v >= self.lm:
                vocabulario_fuente_int[k] = suma
                suma += 1
        return vocabulario_fuente_int

    def crearVocabulario(self,texto):
        vocabulario = {}
        for l in texto:
            for palabra in l.split():
                if palabra not in vocabulario:
                    vocabulario[palabra] = 1
                else:
                    vocabulario[palabra] += 1
        return vocabulario

    def limpia(self,texto):
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

    def vocabularioObjetivo(self,vocabulario):
        vocabulario_objetivo_int = {}

        suma = 0
        for k, v in vocabulario.items():
            if v >= self.lm:
                vocabulario_objetivo_int[k] = suma
                suma += 1
        return vocabulario_objetivo_int

    def palabrasEnteros(self,texto_fuente, vocabulario_fuente_int):
        fuente_int = []
        for l in texto_fuente:
            s = []
            for p in l.split():
                if p not in vocabulario_fuente_int:
                    s.append(vocabulario_fuente_int['<UNK>'])
                else:
                    s.append(vocabulario_fuente_int[p])
            fuente_int.append(s)
        return fuente_int

    def objetivoInt(self,texto_objetivo, vocabulario_objetivo_int):
        objetivo_int = []
        for l in texto_objetivo:
            s = []
            for p in l.split():

                if p not in vocabulario_objetivo_int:
                    s.append(vocabulario_objetivo_int['<UNK>'])
                else:
                    s.append(vocabulario_objetivo_int[p])
            objetivo_int.append(s)
        return objetivo_int


