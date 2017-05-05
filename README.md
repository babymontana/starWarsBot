# starWarsBot

# Instrucciones

Los subtitulos de las peliculas estan en el archivo starwars.txt

En el archivo main.py se ejecuta el pre-procesamiento de los datos , creacion del modelo de la red neuronal y entrenamiento de los datos.

Dentro de la clase modelo se crea la red neuronal y se entrena mediante el método entranar, en el metodo recibe los vocabularios de el preprocesamiento del data set, para hacerlo funcioar se  configuraron algunos valores para poder entrenar la red:

        learning_rate_decay = 0.95
        display_step = 50
        stop_early = 0
        stop = 3
        total_train_loss = 0
        summary_valid_loss = []

posteriormente se modificó la linea de codigo 202, originalmente el codigo tenia:

batch_i % 235 ==0 pero lo cambie a  batch_i % 100 ==0

por que de lo contrario nunca guardaba el modelo puesto que batch_i nunca llegaba a 235.

En el archivo log.txt esta la bitacora del entrenamiento, en realidad no logramos entrenar por muchas epocas (unicamente 9) puesto que la perdida no disminuia lo suficiente y aun asi tarda varias horas.

# Probar Bot

Despues de entrenar el bot se crearon tres archivos:

* modelo.ckpt.index
* modelo.ckpt.meta 

y un último archivo : modelo.ckpt.data-00000-of-000001 que NO se encuenra en el git por que pesa mucho, aqui esta la liga => https://drive.google.com/open?id=0B6gBuCcx5wr3bUlTUTF0dVJjSlk

para probarlo hay que correr el archivo bot.py y asignar en la variable input_sentence la entrada de texto al bot

# Pendiente

Actualmente existe el problema que el bot solo contesta una oración simple aun y con muchas horas de entrenamiento por lo que debemos encontrar la forma que conteste de mejor forma.

