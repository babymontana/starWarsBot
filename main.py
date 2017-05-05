from procesadorTexto import ProcesadorTexto
from modelo import Modelo



def main():
    procesa = ProcesadorTexto()
    model = Modelo(procesa.lm,procesa.vocabulario_fuente_int,procesa.vocabulario_objetivo_int,procesa.fuente_int, procesa.objetivo_int)
    model.entrenar()

if __name__ =='__main__':
    main()