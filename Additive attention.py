from tensorflow.keras.layers import Dense, Lambda, Dot, Activation, Concatenate
from tensorflow.keras.layers import Layer
import random
from datetime import datetime
from tensorflow.keras.layers import Reshape

class addAttention(Layer):
    static_var = 0

    def __init__(self, units, **kwargs):
        self.units = units
        super().__init__(**kwargs)
        Attention.static_var +=1

    def __call__(self, inputs):
        """
        Many-to-one attention mechanism for Keras.
        @param inputs: 3D tensor with shape (batch_size, time_steps, input_dim).
        @return: 2D tensor with shape (batch_size, units)
        """
        hidden_states = inputs
        # print(hidden_states.shape())
        hidden_size = int(hidden_states.shape[2])
        score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec'+str(self.static_var))(hidden_states)
        #            score_first_part           dot        last_hidden_state     => attention_weights
        h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state'+str(self.static_var))(hidden_states)
        score = Dot(axes=[1, 2], name='attention_score'+str(self.static_var))([h_t, score_first_part])
        attention_weights = Activation('softmax', name='attention_weight'+str(self.static_var))(score)
        # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
        context_vector = Dot(axes=[1, 1], name='context_vector'+str(self.static_var))([hidden_states, attention_weights])
        pre_activation = Concatenate(name='attention_output'+str(self.static_var))([context_vector, h_t])
        attention_vector = Dense(self.units, use_bias=False, activation='tanh', name='attention_vector'+str(self.static_var))(pre_activation)
        attention_vector = Reshape((self.units,1))(attention_vector)
        return attention_vector

    def get_config(self):
        return {'units': self.units}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
