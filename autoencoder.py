from keras import Input, Model
from keras.layers import Bidirectional, LSTM, Conv1D, Dense, TimeDistributed, concatenate,\
    Multiply, RepeatVector, Highway, Embedding, Reshape
from keras.optimizers import Adam

def build_autoencoder(max_len=64, max_features=39, char_emb_dim=20,embedding_dim = 128,
                      filters={2: 16, 3: 8, 4: 8}, dropout=0.5, optimizer=Adam, lr=0.001,
                      loss = 'categorical_crossentropy'):
    input_layer = Input(shape=(max_len,), dtype='int32')

    char_embeddings_layer = Embedding(max_features, char_emb_dim, input_length=max_len)(input_layer)

    cnn_encoder_layers = []
    for support, n_filt in filters.items():
        encoder_conv = Conv1D(n_filt, support, border_mode='same', activation='relu')(char_embeddings_layer)
        cnn_encoder_layers.append(encoder_conv)
    cnn_encoder = concatenate(cnn_encoder_layers)

    highway_encoder = TimeDistributed(Highway(activation='relu'))(cnn_encoder)

    n_filters = sum(filters.values())
    lstm_encoder = Bidirectional(LSTM(n_filters // 2, dropout=dropout, return_state=True,
                                      return_sequences=False))
    lstm_encoder, forward_h, forward_c, backward_h, backward_c = lstm_encoder(highway_encoder)
    lstm_encoder_repeated = RepeatVector(max_len)
    lstm_encoder_input = lstm_encoder_repeated(lstm_encoder)

    lstm_encoder_input = Multiply()([lstm_encoder_input, highway_encoder])

    lstm_decoder = Bidirectional(LSTM(embedding_dim // 2, dropout=dropout, return_state=False,
                                      return_sequences=False))
    lstm_decoder = lstm_decoder(lstm_encoder_input)

    encoder_last_layer = lstm_decoder
    dense_decoder = Dense(max_len * char_emb_dim, name='decoder_begin')(
        lstm_decoder)
    dense_decoder = Reshape((max_len, char_emb_dim))(dense_decoder)

    highway_decoder = TimeDistributed(Highway(activation='relu'))(dense_decoder)

    cnn_decoders_layers = []
    for support, n_filt in filters.items():
        decoder_conv = Conv1D(n_filt, support, border_mode='same', activation='relu')(highway_decoder)
        cnn_decoders_layers.append(decoder_conv)
    cnn_decoder = concatenate(cnn_decoders_layers)
    autoencoder_output = TimeDistributed(Dense(max_features, activation='softmax'), name='decoder_end')(
        cnn_decoder)

    autoencoder = Model(input=input_layer, output=autoencoder_output)
    autoencoder.compile(optimizer(lr), loss)
    autoencoder.summary()
    encoder = Model(inputs=input_layer, outputs=encoder_last_layer, name='encoder')
    return autoencoder, encoder

build_autoencoder()
