import keras
from keras_self_attention import SeqWeightedAttention
from keras.layers import Bidirectional


def new_architecture(x_len, vocab_size, layers, bi, att, cells):
    inputs = keras.layers.Input(
        shape=(x_len, vocab_size,), name='Input')

    prev = inputs
    for i in range(layers):
        ret_seq = True
        if i == layers-1 and att == False:
            ret_seq = False

        this_layer = keras.layers.LSTM(
            cells,
            dropout=0.4,
            name='LSTM_%s' %i,
            return_sequences=ret_seq
        )
        if bi:
            this_layer = Bidirectional(
                this_layer,
                name='bi_%s' %i
            )
        prev = this_layer(prev)

    if att:
        attention = SeqWeightedAttention(
            return_attention=False,
            name='Attention'
        )
        prev = attention(prev)

    dense = keras.layers.Dense(
        vocab_size, activation='softmax', name="dense_outputs")(prev)

    model = keras.Model(inputs=inputs, outputs=[dense])

    # lstm = Bidirectional(
    #     keras.layers.LSTM(
    #         cells,
    #         dropout=0.4,
    #         name="LSTM",
    #         return_sequences=True),
    #     name="bi1"
    # )(inputs)

    # lstm2 = Bidirectional(
    #     keras.layers.LSTM(
    #         cells,
    #         dropout=0.4,
    #         name="LSTM2",
    #         return_sequences=True),
    #     name="bi2"
    # )(lstm)
    return model
