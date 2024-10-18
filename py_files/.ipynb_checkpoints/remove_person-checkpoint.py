import pickle
import sys

def get_names_encodings(file_path):
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)
    
def save_data(data, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
names = get_names_encodings('../data/processed/lfw_names.pickle')
encodings = get_names_encodings('../data/processed/lfw_encodings.pickle')

def main():
    args = sys.argv[1:]
    name_to_remove = args[0]
    if (name_to_remove in names):
        for i, n in enumerate(names):
            if n == name_to_remove:
                names.pop(i)
                encodings.pop(i)
                remove_names_encodings(name_to_remove)
    else:
        save_data(names, '../data/processed/lfw_names.pickle')
        save_data(encodings, '../data/processed/lfw_encodings.pickle')
        
if __name__ == '__main__':
    main()