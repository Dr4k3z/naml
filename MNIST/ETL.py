import numpy as np
from PIL import Image

class ETL:
    def __init__(self,path):
        self.__path = "./"+path

    def loadData(self,reduce=True):
        with np.load(self.__path) as f:
            X_train, y_train = f['x_train'], f['y_train']
            X_test, y_test = f['x_test'], f['y_test']

            # For performance, training and testing set size is reduced
            if reduce:
                max_row_training = 5000
                max_row_testing = 10000
                X_train = X_train[:max_row_training] #reduce sample for performance
                y_train = y_train[:max_row_training]
                X_test = X_test[:max_row_testing]
                y_test = y_test[:max_row_testing] #recude sample for performance
            
            # Flatten the images (28x28) to 1D (784)
            X_train = X_train.reshape((X_train.shape[0], -1))
            X_test = X_test.reshape((X_test.shape[0],-1))

            label_train = np.zeros((y_train.shape[0],10))
            label_train[np.arange(y_train.shape[0]),y_train] = 1

            label_test = np.zeros((y_test.shape[0],10))
            label_test[np.arange(y_test.shape[0]),y_test] = 1

            return X_train, label_train, X_test, label_test

    @staticmethod        
    def readImg(name):
        img = Image.open(name).convert("L")
        return np.array(img).flatten().reshape(1,-1)

# thx chatgpt
class NumpyDataLoader:
    def __init__(self, features, labels, batch_size, shuffle=True, drop_last=False):
        assert len(features) == len(labels), "Features and labels must have the same length"
        self.features = features
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.indices = np.arange(len(self.features))
    
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.current_index = 0
        return self
    
    def __next__(self):
        if self.current_index >= len(self.features):
            raise StopIteration
        
        batch_indices = self.indices[self.current_index:self.current_index + self.batch_size]
        
        # If drop_last is True, and the last batch is smaller than batch_size, discard it
        if self.drop_last and len(batch_indices) < self.batch_size:
            raise StopIteration
        
        batch_features = self.features[batch_indices]
        batch_labels = self.labels[batch_indices]
        self.current_index += self.batch_size
        return batch_features, batch_labels
    
    def __len__(self):
        # If drop_last is True, ignore the last incomplete batch
        if self.drop_last:
            return len(self.features) // self.batch_size
        return (len(self.features) + self.batch_size - 1) // self.batch_size

import matplotlib.pyplot as plt    

if __name__ == "__main__":
    etl = ETL("mnist.npz",reduce=False)
    X_train,y_train,X_test,y_test = etl.loadData()

    batch_size = 128
    train_loader = NumpyDataLoader(X_train, y_train, batch_size, shuffle=True, drop_last=True)

    for i,(img,lbl) in enumerate(train_loader):
        print(i,img.shape,lbl.shape)