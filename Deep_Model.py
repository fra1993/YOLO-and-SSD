import tensorflow as tf

def Build_model(initializer_, input_image_size, BB_per_cell, Classes, batch_, model_name):
    
    # input lauer
    input_layer = tf.keras.layers.Input(shape=input_image_size, batch_size=batch_, name="input")

    # down_sampling
    conv_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1),
                 activation='relu', kernel_initializer=initializer_, padding="same", name="conv_1")(input_layer)

    pooling_1 = tf.keras.layers.MaxPool2D(pool_size=(2,2), name="pool_1")(conv_1)
    
    batch_norm_1 = tf.keras.layers.BatchNormalization()(pooling_1)

    conv_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1),
                 activation='relu', kernel_initializer=initializer_,padding="same", name="conv_2")(batch_norm_1)
    
    pooling_2 = tf.keras.layers.MaxPool2D(pool_size=(2,2), name="pool_2")(conv_2)
    
    batch_norm_2 = tf.keras.layers.BatchNormalization()(pooling_2)

    conv_3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1),
                 activation='relu', kernel_initializer=initializer_,padding="same", name="conv_3")(batch_norm_2)
    
    pooling_3 = tf.keras.layers.MaxPool2D(pool_size=(2,2), name="pool_3")(conv_3)
    
    batch_norm_3 = tf.keras.layers.BatchNormalization()(pooling_3)

    conv_4 = tf.keras.layers.Conv2D(filters=BB_per_cell*(5+Classes), kernel_size=(3,3), strides=(1,1),
                 activation='relu', kernel_initializer=initializer_,padding="same", name="conv_4")(batch_norm_3)
    
    pooling_4 = tf.keras.layers.MaxPool2D(pool_size=(2,2), name="detection_1")(conv_4)
    
    batch_norm_4 = tf.keras.layers.BatchNormalization()(pooling_4)

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
    
    return tf.keras.Model(inputs=input_layer, outputs=[pooling_4], name=model_name)