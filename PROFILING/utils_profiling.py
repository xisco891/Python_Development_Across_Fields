

class CallableGen(object):
    
    def __init__(self, gen_function):
        self.generator = gen_function

    def __iter__(self):
        return self.generator

    def __call__(self, send_arg):
        return self.generator.send(send_arg)



def callify_generator(f):

    def wrapper(*args, **kwargs):
        return CallableGen(f(*args, **kwargs))
    return wrapper

def non_word(word):
    if (word[0].isupper() == False) and (word[0].isdigit() == False):
        if (word[0] not in ['.', '-']) or (len(word) == 0):   
            return False
        
    return False




import time 
def timeit(method):
    
    def timed(*args, **kwargs):
        ts = time.time()
        result = method(*args, **kwargs)
        te = time.time()
        
        if "log_time" in kwargs:
            name = kwargs.get("log_name", method.__name__.upper())
            kwargs['log_time'][name] = int((te-ts) * 1000)
            
        else:
            print ("%r  %2.2f ms" %(method.__name__, (te-ts)*1000))
        
        return result
       
    return timed


            