import pickle 

def write_pickle(var, filename):
    '''Write variable to pickle file with given filename'''
    with open(f'{filename}', 'wb') as f:
        pickle.dump(var, f)
        print(f'Successfully saved as {filename}')

def read_pickle(filename):
    '''Read variable from pickle file with given filename'''
    with open(f'{filename}', 'rb') as f:
        var = pickle.load(f)
        return var