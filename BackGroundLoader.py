from threading import Thread 

def BackGroundLoader(generator, batch_size):
    
    '''Apparently doesnt work
    '''
    
    # generate an ampty list of fixed shape
    dataBuffer = [None]*batch_size
    
    #define the target function to run in the backgroud
    def LoadNextBatch(generator,dataBuffer,batch_size):
        for idx in range(batch_size):
            dataBuffer[idx] = generator.__next__()
    
    # lauch the thread
    loader = Thread(target=LoadNextBatch, args=(generator,dataBuffer,batch_size))
    loader.start()
    
    while True:
        
        # wait until the previos thread has finished the loading
        loader.join()
        
        # trannsform the batch in a Tensorflow-Readable format
        batch = dataBuffer
        
        # start next thread
        loader = Thread(target=LoadNextBatch, args=(generator,dataBuffer,batch_size))
        loader.start()
        
    yield batch