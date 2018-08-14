from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM,concatenate)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization(name='simp_rnn')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = GRU(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name='simp_rnn')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    layers=[]
    if recur_layers ==1:
        rnn = GRU(units=units,return_sequences=True,implementation=2)(input_data)
        rnn = BatchNormalization()(rnn)
    else:
        for i in range(recur_layers):
            rnn_i = GRU(units=units,return_sequences=True,implementation=2)(input_data)
            rnn_i = BatchNormalization()(rnn_i)
            layers.append(rnn_i)
        rnn = concatenate(layers) 
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(GRU(units,return_sequences=True,implementation=2,name='bi_rnn'),merge_mode='concat')(input_data)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def cnn_bidirectional_rnn_model(input_dim,filters,kernel_size,conv_stride,conv_border_mode,units,output_dim=29):
    """Build a cnn-bidirectional_rnn network for speech
    """
    #Main acoustic input
    input_data = Input(name='the_input',shape=(None,input_dim))
    ## Adding first CNN1D layer
    cnn = Conv1D(filters,kernel_size
           ,strides = conv_stride
           ,padding=conv_border_mode
           ,activation='relu'
           ,name='conv1d')(input_data)
    ## Adding batch_normalization
    b_cnn = BatchNormalization(name='b_cnn')(cnn)
    ##Adding bidirectional rnn
    bi_rnn = Bidirectional(GRU(units,return_sequences=True,activation='relu',implementation=2,name='bi_rnn'),merge_mode='concat')(b_cnn)
    ##Adding batch_normalization
    b_birnn = BatchNormalization(name='b_birnn')(bi_rnn)
    ##Adding TimeDistributed layer
    time_dist = TimeDistributed(Dense(output_dim))(b_birnn)
    ## Activation layer
    y_pred = Activation('softmax',name='softmax')(time_dist)
    #sepcify model
    model = Model(inputs = input_data,outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(x,kernel_size,conv_border_mode,conv_stride)
    print(model.summary())
    return model

def deep_bidirectional_rnn(input_dim,units,recur_layers,output_dim=29):
    """Build deep birectional rnn 
    """
    ## Main acoustic input layer
    input_data = Input(name='the_input',shape = (None,input_dim))
    ##Deep birectional layer
    layers = []
    if recur_layers == 1:
        bi_rnn = Bidirectional(GRU(units,return_sequences=True,implementation=2,activation='relu'),merge_mode = 'concat')(input_data)
        b_birnn = BatchNormalization()(bi_rnn)
    else:
        for i in range(recur_layers):
            bi_rnn_i = Bidirectional(GRU(units,return_sequences=True,implementation=2,activation='relu'),merge_mode='concat')(input_data)
            bi_rnn_i = BatchNormalization()(bi_rnn_i)
            layers.append(bi_rnn_i)
        rnn = concatenate(layers,axis=1)
    ## Add Timedistributed layer
    time_dist = TimeDistributed(Dense(output_dim))(rnn)
    ##Activation layer
    y_pred = Activation('softmax',name='softmax')(time_dist)
    #Specify model
    model = Model(inputs=input_data,outputs=y_pred)
    model.output_length = lambda x:x
    print(model.summary())
    return model
                      

def final_model(input_dim,filters,kernel_size,conv_stride,conv_border_mode,units,output_dim=29):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Specify the layers in your network
    simp_CNN = Conv1D(filters,kernel_size,strides=conv_stride,padding=conv_border_mode,activation='relu',name='CNN_model')(input_data)
    # Add batch normaliation
    bn_CNN = BatchNormalization(name='bn_CNN')(simp_CNN)
    # ADD bidirectional GRU with dropout
    bi_RNN = Bidirectional(GRU(units,return_sequences=True,implementation=2,recurrent_dropout=0.5,dropout=0.5,
                               name='bi_RNN'),merge_mode='concat')(bn_CNN)
    # ADD batch normalization
    bn_RNN = BatchNormalization(name='bn_RNN')(bi_RNN)
    # ADD time distributed layers
    time_dist = TimeDistributed(Dense(output_dim))(bn_RNN)
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax',name='softmax')(time_dist)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    model.output_length = lambda x: cnn_output_length(x,kernel_size,conv_border_mode,conv_stride)
    print(model.summary())
    return model

def dilated_CNN_model(input_dim,filters,kernel_size,conv_stride,conv_border_mode,output_dim=29):
    """ Build a dialated CNN model or speech
    """
    # Main acoustic input
    input_data = Input(name='the_input',shape=(None,input_dim))
    # CNN layer with normal convolution
    CNN_1 = Conv1D(filters,kernel_size,strides=conv_stride,padding='causal',dilation_rate=1,
                   activation='relu',name='CNN_1')(input_data)
    CNN_2 = Conv1D(filters,kernel_size,strides=1,padding='causal',dilation_rate=2,
                   activation='relu',name='CNN_2')(CNN_1)
    CNN_4 = Conv1D(filters,kernel_size,strides=1,padding='causal',dilation_rate=4,
                   activation='relu',name='CNN_4')(CNN_2)
    CNN_8 = Conv1D(filters,kernel_size,strides=1,padding='causal',dilation_rate=8,
                   activation='relu',name='CNN_8')(CNN_4)
    # Time distributed dense layer
    time_dist = TimeDistributed(Dense(output_dim))(CNN_8)
    ## Add softmax layer
    y_pred = Activation('softmax',name='softmax')(time_dist)
    ## sepcify model
    model = Model(inputs=input_data,outputs=y_pred)
    ##specify model output_length
    model.output_length = lambda x: cnn_output_length(x,kernel_size,conv_border_mode,conv_stride)
    print(model.summary())
    return model