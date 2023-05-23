import tensorflow as tf
import numpy as np


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, dropout_rate):
        super(TransformerEncoder, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate

        self.embedding = tf.keras.layers.Dense(d_model, activation='relu')
        self.positional_encoding = self.positional_encoding(maximum_position_encoding=10000,
                                                             d_model=d_model)
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.enc_layers = [self.EncoderLayer(d_model, num_heads, dff, dropout_rate)
                           for _ in range(num_layers)]

    def call(self, x, training):
        seq_len = tf.shape(x)[1]

        # Adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)

        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        print(self.positional_encoding[:, :seq_len, :])
        x += self.positional_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training)

        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_layers': self.num_layers,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'dropout_rate': self.dropout_rate
        })
        return config

    def EncoderLayer(self, d_model, num_heads, dff, dropout_rate=0.2):
        inputs = tf.keras.Input(shape=(None, d_model))
        attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(inputs, inputs)
        attention = tf.keras.layers.Dropout(dropout_rate)(attention)
        attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention)
        outputs = tf.keras.layers.Dense(units=dff, activation='relu')(attention)
        outputs = tf.keras.layers.Dense(units=d_model)(outputs)
        outputs = tf.keras.layers.Dropout(dropout_rate)(outputs)
        outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + outputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs, name='EncoderLayer')

    def positional_encoding(self, maximum_position_encoding, d_model):
        angle_rates = 1 / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
        angle_rads = np.arange(maximum_position_encoding)[:, np.newaxis] * angle_rates

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]
        # pos_encoding = np.reshape(pos_encoding, (pos_encoding.shape[1], pos_encoding.shape[2]))
        return tf.cast(pos_encoding, dtype=tf.float32)


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim),]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

#  Layer (type)                Output Shape              Param #   
# =================================================================
#  input_1 (InputLayer)        [(None, 68)]              0         
                                                                 
#  token_and_position_embeddin  (None, 68, 32)           75424     
#  g (TokenAndPositionEmbeddin                                     
#  g)                                                              
                                                                 
#  transformer_block (Transfor  (None, 68, 32)           44192     
#  merBlock)                                                       
                                                                 
#  global_average_pooling1d (G  (None, 32)               0         
#  lobalAveragePooling1D)                                          
                                                                 
#  dropout_2 (Dropout)         (None, 32)                0         
                                                                 
#  dense_2 (Dense)             (None, 1144)              37752     
                                                                 
#  dropout_3 (Dropout)         (None, 1144)              0         
                                                                 
#  dense_3 (Dense)             (None, 2289)              2620905   