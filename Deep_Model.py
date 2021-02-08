import tensorflow as tf

def mini_conv_block(input_, filters, kernel_size, strides, activation, ker_initializer, padd_):
    batch_norm = tf.keras.layers.BatchNormalization()(input_)
    conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                 activation=activation, kernel_initializer=ker_initializer, padding=padd_)(batch_norm)
    pooling = tf.keras.layers.MaxPool2D(pool_size=(2,2))(conv)
    return pooling 
    

def Build_model(initializer_, input_image_size, BB_per_cell, Classes, batch_, model_name):
    
    # input lauer
    input_layer = tf.keras.layers.Input(shape=input_image_size, batch_size=batch_, name="input")

    first_mini_block = mini_conv_block(input_layer, filters=512, kernel_size=(3,3), 
                       strides=(1,1), activation="elu", ker_initializer=initializer_, padd_="same")

    second_mini_block = mini_conv_block(first_mini_block, filters=256, kernel_size=(3,3), 
                       strides=(1,1), activation="elu", ker_initializer=initializer_, padd_="same")
    
    third_mini_block = mini_conv_block(second_mini_block, filters=128, kernel_size=(3,3), 
                       strides=(1,1), activation="elu", ker_initializer=initializer_, padd_="same")
    
    fourth_mini_block = mini_conv_block(third_mini_block, filters=64, kernel_size=(3,3), 
                       strides=(1,1), activation="elu", ker_initializer=initializer_, padd_="same")
    
    fifth_mini_block = mini_conv_block(fourth_mini_block, filters=64, kernel_size=(3,3), 
                       strides=(1,1), activation="elu", ker_initializer=initializer_, padd_="same")
    
    batch_norm_3 = tf.keras.layers.BatchNormalization()(fifth_mini_block)

    conv_4 = tf.keras.layers.Conv2D(filters=BB_per_cell*(5+Classes), kernel_size=(3,3), strides=(1,1),
                 activation='sigmoid',padding="same", name='detection_sigmoid')(batch_norm_3)

    # up sampling

    # up_sampling_1 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="nearest", name="up_sampling_1")(pooling_4)
    # 
    # concat_layer_1 = tf.keras.layers.Concatenate(name="detection_2")([conv_4,up_sampling_1])
    # 
    # conv_5 = tf.keras.layers.Conv2D(filters=BB_per_cell*(5+Classes), kernel_size=(1,1), strides=(1,1),
    #              activation='relu', kernel_initializer=relu_initializer, name="detection_2")(up_sampling_1)
    # 
    # up_sampling_2 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="nearest", name="up_sampling_2")(conv_5)
    # 
    # conv_6 = tf.keras.layers.Conv2D(filters=BB_per_cell*(5+Classes), kernel_size=(1,1), strides=(1,1),
    #              activation='relu', kernel_initializer=relu_initializer, name="detection_3")(up_sampling_2)
    # 
    
    return tf.keras.Model(inputs=input_layer, outputs=[conv_4], name=model_name)