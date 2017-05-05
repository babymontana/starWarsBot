import tensorflow as tf
import numpy as np
import time

class Modelo:

    epochs = 100
    batch_size = 128
    rnn_size = 512
    num_layers = 2
    encoding_embedding_size = 512
    decoding_embedding_size = 512
    attn_length = 10
    learning_rate = 0.0005
    keep_probability = 0.8

    def __init__(self,lm,vocabulario_fuente_int,vocabulario_objetivo_int,fuente_int, objetivo_int):

        self.lm = lm
        self.vocabulario_fuente_int= vocabulario_fuente_int
        self.vocabulario_objetivo_int=vocabulario_objetivo_int
        self.grafo = tf.Graph()
        with  self.grafo.as_default():

            # Carga datos del modelo
            self.input_data, self.targets, self.lr, self.keep_prob = self.datos_modelo()
            # Tamaño de la oración será el tamaño del batch
            self.sequence_length = tf.placeholder_with_default(self.lm, None, name='sequence_length')
            input_shape = tf.shape(self.input_data)

            # Creación de las probabilidades del modelo
            train_logits, inference_logits = self.seq2seq_model(
                tf.reverse(self.input_data, [-1]), self.targets, self.keep_prob, self.batch_size, self.sequence_length,
                len(self.vocabulario_fuente_int),
                len(self.vocabulario_objetivo_int), self.encoding_embedding_size, self.decoding_embedding_size, self.rnn_size, self.num_layers,
                self.vocabulario_objetivo_int, self.attn_length)

            # Creación de los tensores
            tf.identity(inference_logits, 'logits')
            with tf.name_scope("optimization"):
                # función de pérdida
                self.cost = tf.contrib.seq2seq.sequence_loss(
                    train_logits,
                    self.targets,
                    tf.ones([input_shape[0], self.sequence_length]))

                # Optimizador
                optimizer = tf.train.AdamOptimizer(self.learning_rate)

                # Gradient Descend
                gradients = optimizer.compute_gradients(self.cost)
                capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if
                                    grad is not None]
                self.train_op = optimizer.apply_gradients(capped_gradients)

                entrenamiento_split = int(len(fuente_int) * 0.1)

                self.entrenamiento_fuente = fuente_int[entrenamiento_split:]
                self.entrenamiento_objetivo = objetivo_int[entrenamiento_split:]

                self.fuente_valido = fuente_int[:entrenamiento_split]
                self.objetivo_valido = objetivo_int[:entrenamiento_split]

    def datos_modelo(self):
        input_data = tf.placeholder(tf.int32, [None, None], name='input')
        targets = tf.placeholder(tf.int32, [None, None], name='targets')
        lr = tf.placeholder(tf.float32, name='learning_rate')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        return input_data, targets, lr, keep_prob

    def codificacion_input(self,target_data, vocab_to_int, batch_size):
        ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
        dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)
        return dec_input

    def codificacion_capa(self,rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length, attn_length):
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob)
        cell = tf.contrib.rnn.AttentionCellWrapper(drop, attn_length, state_is_tuple=True)
        enc_cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)
        _, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=enc_cell,
                                                       cell_bw=enc_cell,
                                                       sequence_length=sequence_length,
                                                       inputs=rnn_inputs,
                                                       dtype=tf.float32)

        return enc_state

    def decodificacion_capa_entrenamiento(self,encoder_state, dec_cell, dec_embed_input, sequence_length, decoding_scope,
                                           output_fn, keep_prob):
        train_decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_train(encoder_state)
        train_pred, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
            dec_cell, train_decoder_fn, dec_embed_input, sequence_length, scope=decoding_scope)
        train_pred_drop = tf.nn.dropout(train_pred, keep_prob)
        return output_fn(train_pred_drop)

    def decodificacion_capa_inferencia(self,encoder_state, dec_cell, dec_embeddings, start_of_sequence_id,
                                       end_of_sequence_id,
                                       maximum_length, vocab_size, decoding_scope, output_fn, keep_prob):
        infer_decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_inference(
            output_fn, encoder_state, dec_embeddings, start_of_sequence_id, end_of_sequence_id, maximum_length,
            vocab_size)
        infer_logits, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(dec_cell, infer_decoder_fn, scope=decoding_scope)
        return infer_logits

    def decodificacion_capa(self,dec_embed_input, dec_embeddings, encoder_state, vocab_size, sequence_length, rnn_size,
                            num_layers, vocab_to_int, keep_prob, attn_length):
        with tf.variable_scope("decoding") as decoding_scope:
            lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
            drop = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob)
            cell = tf.contrib.rnn.AttentionCellWrapper(drop, attn_length, state_is_tuple=True)
            dec_cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)

            weights = tf.truncated_normal_initializer(stddev=0.1)
            biases = tf.zeros_initializer()
            output_fn = lambda x: tf.contrib.layers.fully_connected(x,
                                                                    vocab_size,
                                                                    None,
                                                                    scope=decoding_scope,
                                                                    weights_initializer=weights,
                                                                    biases_initializer=biases)

            train_logits = self.decodificacion_capa_entrenamiento(
                encoder_state[0], dec_cell, dec_embed_input, sequence_length, decoding_scope, output_fn, keep_prob)
            decoding_scope.reuse_variables()
            infer_logits =self. decodificacion_capa_inferencia(encoder_state[0], dec_cell, dec_embeddings,
                                                          vocab_to_int['<GO>'],
                                                          vocab_to_int['<EOS>'], sequence_length, vocab_size,
                                                          decoding_scope, output_fn, keep_prob)

        return train_logits, infer_logits

    def seq2seq_model(self, input_data, target_data, keep_prob, batch_size, sequence_length, source_vocab_size,
                      target_vocab_size,
                      enc_embedding_size, dec_embedding_size, rnn_size, num_layers, vocab_to_int, attn_length):
        enc_embed_input = tf.contrib.layers.embed_sequence(input_data, source_vocab_size + 1, enc_embedding_size)
        enc_state = self.codificacion_capa(enc_embed_input, rnn_size, num_layers, keep_prob, sequence_length, attn_length)

        dec_input = self.codificacion_input(target_data, vocab_to_int, batch_size)
        dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size + 1, dec_embedding_size], -1.0, 1.0))
        dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

        train_logits, infer_logits = self.decodificacion_capa(dec_embed_input, dec_embeddings, enc_state,
                                                         target_vocab_size + 1,
                                                         sequence_length, rnn_size, num_layers, vocab_to_int, keep_prob,
                                                         attn_length)

        return train_logits, infer_logits

    def pad_sentence_batch(self,sentence_batch, vocab_to_int):
        max_sentence = max([len(sentence) for sentence in sentence_batch])
        return [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]

    def batch_data(self,source, target, batch_size):
        for batch_i in range(0, len(source) // batch_size):
            start_i = batch_i * batch_size
            source_batch = source[start_i:start_i + batch_size]
            target_batch = target[start_i:start_i + batch_size]
            yield (np.array(self.pad_sentence_batch(source_batch, self.vocabulario_fuente_int)),
                   np.array(self.pad_sentence_batch(target_batch, self.vocabulario_objetivo_int)))

    def entrenar(self):

        learning_rate_decay = 0.95
        display_step = 50
        stop_early = 0
        stop = 3
        total_train_loss = 0
        summary_valid_loss = []

        checkpoint = "modelo.ckpt"

        with tf.Session(graph=self.grafo) as sess:
            sess.run(tf.global_variables_initializer())

            for epoch_i in range(1, self.epochs + 1):
                for batch_i, (source_batch, target_batch) in enumerate(
                        self.batch_data(self.entrenamiento_fuente, self.entrenamiento_objetivo, self.batch_size)):
                    start_time = time.time()
                    _, loss = sess.run(
                        [self.train_op, self.cost],
                        {self.input_data: source_batch,
                         self.targets: target_batch,
                         self.lr: self.learning_rate,
                         self.sequence_length: target_batch.shape[1],
                         self.keep_prob: self.keep_probability})

                    total_train_loss += loss
                    end_time = time.time()
                    batch_time = end_time - start_time
                    print(batch_i)
                    if batch_i % display_step == 0:
                        print('Epoca {:>3}/{} Batch {:>4}/{} - Perdida: {:>6.3f}, Segundos Procesamiento: {:>4.2f}'
                              .format(epoch_i,
                                      self.epochs,
                                      batch_i,
                                      len(self.entrenamiento_fuente) // self.batch_size,
                                      total_train_loss / display_step,
                                      batch_time * display_step))
                        total_train_loss = 0

                    if batch_i % 100 == 0 and batch_i > 0:
                        total_valid_loss = 0
                        start_time = time.time()
                        for batch_ii, (source_batch, target_batch) in \
                                enumerate(self.batch_data(self.fuente_valido, self.objetivo_valido, self.batch_size)):
                            valid_loss = sess.run(
                                self.cost, {self.input_data: source_batch,
                                       self.targets: target_batch,
                                       self.lr: self.learning_rate,
                                       self.sequence_length: target_batch.shape[1],
                                       self.keep_prob: 1})
                            total_valid_loss += valid_loss
                        end_time = time.time()
                        batch_time = end_time - start_time
                        avg_valid_loss = total_valid_loss / (len(self.fuente_valido) / self.batch_size)
                        print('Perdida Válida: {:>6.3f}, Segundos: {:>5.2f}'.format(avg_valid_loss, batch_time))

                        self.learning_rate *= learning_rate_decay

                        summary_valid_loss.append(avg_valid_loss)
                        if avg_valid_loss <= min(summary_valid_loss):
                            print('Nuevo CheckPoint!')
                            stop_early = 0
                            saver = tf.train.Saver()
                            saver.save(sess, checkpoint)

                        else:
                            print("Sin Mejora")
                            stop_early += 1
                            if stop_early == stop:
                                break
                if stop_early == stop:
                    print("Termina entrenamiento.")
                    break
