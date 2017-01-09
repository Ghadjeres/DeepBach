from keras.engine import Input
from keras.engine import Model
from keras.engine import merge
from keras.layers import Dense, TimeDistributed, LSTM, Dropout, Activation, Lambda


def deepBach(num_features_lr, num_features_c, num_pitches, num_features_meta, num_units_lstm=[200],
             num_dense=200, timesteps=16):
    """

    :param num_features_lr: size of left or right features vectors
    :param num_features_c: size of central features vectors
    :param num_pitches: size of output
    :param num_units_lstm: list of lstm layer sizes
    :param num_dense:
    :return:
    """
    # input
    left_features = Input(shape=(timesteps, num_features_lr), name='left_features')
    right_features = Input(shape=(timesteps, num_features_lr), name='right_features')
    central_features = Input(shape=(num_features_c,), name='central_features')
    # input metadatas
    left_metas = Input(shape=(timesteps, num_features_meta), name='left_metas')
    right_metas = Input(shape=(timesteps, num_features_meta), name='right_metas')
    central_metas = Input(shape=(num_features_meta,), name='central_metas')

    # embedding layer for left and right
    embedding_left = Dense(input_dim=num_features_lr + num_features_meta,
                           output_dim=num_dense, name='embedding_left')
    embedding_right = Dense(input_dim=num_features_lr + num_features_meta,
                            output_dim=num_dense, name='embedding_right')

    predictions_left = TimeDistributed(embedding_left)(merge((left_features,
                                                              left_metas),
                                                             mode='concat'))
    predictions_right = TimeDistributed(embedding_right)(merge((right_features,
                                                                right_metas),
                                                               mode='concat'))

    predictions_center = merge((central_features, central_metas), mode='concat')

    predictions_center = Dense(num_dense, activation='relu')(predictions_center)
    predictions_center = Dense(num_dense, activation='relu')(predictions_center)

    return_sequences = True
    for k, stack_index in enumerate(range(len(num_units_lstm))):
        if k == len(num_units_lstm) - 1:
            return_sequences = False
        predictions_left = LSTM(num_units_lstm[stack_index],
                                return_sequences=return_sequences,
                                name='lstm_left_' + str(stack_index)
                                )(predictions_left)
        predictions_right = LSTM(num_units_lstm[stack_index],
                                 return_sequences=return_sequences,
                                 name='lstm_right_' + str(stack_index)
                                 )(predictions_right)

    predictions = merge((predictions_left, predictions_center, predictions_right),
                        mode='concat')
    predictions = Dense(num_dense, activation='relu')(predictions)
    pitch_prediction = Dense(num_pitches, activation='softmax',
                             name='pitch_prediction')(predictions)

    model = Model(input=[left_features, central_features, right_features,
                         left_metas, central_metas, right_metas
                         ],
                  output=pitch_prediction)

    model.compile(optimizer='adam',
                  loss={'pitch_prediction': 'categorical_crossentropy'},
                  metrics=['accuracy'])
    model.summary()
    return model


def deepbach_skip_connections(num_features_lr, num_features_c, num_features_meta, num_pitches, num_units_lstm=[200],
                              num_dense=200, timesteps=16):
    """

    :param num_features_lr: size of left or right features vectors
    :param num_features_c: size of central features vectors
    :param num_pitches: size of output
    :param num_units_lstm: list of lstm layer sizes
    :param num_dense:
    :return:
    """
    left_features = Input(shape=(timesteps, num_features_lr), name='left_features')
    right_features = Input(shape=(timesteps, num_features_lr), name='right_features')
    central_features = Input(shape=(num_features_c,), name='central_features')

    left_metas = Input(shape=(timesteps, num_features_meta), name='left_metas')
    right_metas = Input(shape=(timesteps, num_features_meta), name='right_metas')
    central_metas = Input(shape=(num_features_meta,), name='central_metas')

    # embedding layer for left and right
    embedding_left = Dense(input_dim=num_features_lr + num_features_meta,
                           output_dim=num_dense, name='embedding_left')
    embedding_right = Dense(input_dim=num_features_lr + num_features_meta,
                            output_dim=num_dense, name='embedding_right')

    # merge features and metadata
    predictions_left = merge((left_features, left_metas), mode='concat')
    predictions_right = merge((right_features, right_metas), mode='concat')
    predictions_center = merge((central_features, central_metas), mode='concat')

    # input dropout
    predictions_left = Dropout(0.2)(predictions_left)
    predictions_right = Dropout(0.2)(predictions_right)
    predictions_center = Dropout(0.2)(predictions_center)

    # embedding
    predictions_left = TimeDistributed(embedding_left)(predictions_left)
    predictions_right = TimeDistributed(embedding_right)(predictions_right)

    # central NN
    predictions_center = Dense(num_dense, activation='relu')(predictions_center)
    # predictions_center = Dropout(0.5)(predictions_center)
    predictions_center = Dense(num_dense, activation='relu')(predictions_center)

    # left and right recurrent networks
    return_sequences = True
    for k, stack_index in enumerate(range(len(num_units_lstm))):
        if k == len(num_units_lstm) - 1:
            return_sequences = False

        if k > 0:
            # todo difference between concat and sum
            predictions_left_tmp = merge([Activation('relu')(predictions_left), predictions_left_old], mode='sum')
            predictions_right_tmp = merge([Activation('relu')(predictions_right), predictions_right_old], mode='sum')
        else:
            predictions_left_tmp = predictions_left
            predictions_right_tmp = predictions_right

        predictions_left_old = predictions_left
        predictions_right_old = predictions_right
        predictions_left = predictions_left_tmp
        predictions_right = predictions_right_tmp

        predictions_left = LSTM(num_units_lstm[stack_index],
                                return_sequences=return_sequences,
                                name='lstm_left_' + str(stack_index)
                                )(predictions_left)

        predictions_right = LSTM(num_units_lstm[stack_index],
                                 return_sequences=return_sequences,
                                 name='lstm_right_' + str(stack_index)
                                 )(predictions_right)

        # todo dropout here?
        # predictions_left = Dropout(0.5)(predictions_left)
        # predictions_right = Dropout(0.5)(predictions_right)

    # retain only last input for skip connections
    predictions_left_old = Lambda(lambda t: t[:, -1, :],
                                  output_shape=lambda input_shape: (input_shape[0], input_shape[-1])
                                  )(predictions_left_old)
    predictions_right_old = Lambda(lambda t: t[:, -1, :],
                                   output_shape=lambda input_shape: (input_shape[0], input_shape[-1],)
                                   )(predictions_right_old)
    # concat or sum
    predictions_left = merge([Activation('relu')(predictions_left), predictions_left_old], mode='concat')
    predictions_right = merge([Activation('relu')(predictions_right), predictions_right_old], mode='concat')

    predictions = merge([predictions_left, predictions_center, predictions_right],
                        mode='concat')
    predictions = Dense(num_dense, activation='relu')(predictions)
    pitch_prediction = Dense(num_pitches, activation='softmax',
                             name='pitch_prediction')(predictions)

    model = Model(input=[left_features, central_features, right_features,

                         left_metas, right_metas, central_metas],
                  output=pitch_prediction)

    model.compile(optimizer='adam',
                  loss={'pitch_prediction': 'categorical_crossentropy'},
                  metrics=['accuracy'])
    model.summary()
    return model
