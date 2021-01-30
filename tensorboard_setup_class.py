import tensorflow as tf
import os
import time

class tensor_board_setup():
    
    def __init__(self,logs_dir="my_logs"):
        self.root_logdir = os.path.join(os.curdir, logs_dir)
        self.run_id = time.strftime("run%Y_%m_%d-%H_%M_%S")
    
    def setup(self):
        return tf.keras.callbacks.TensorBoard(os.path.join(self.root_logdir,self.run_id))