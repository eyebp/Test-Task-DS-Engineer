from abc import ABC, abstractmethod
import numpy as np
import random
import torch
import pickle

import mnist


class DigitClassificationInterface(ABC):

    @abstractmethod
    def predict(self, image: np.ndarray | torch.Tensor, 
                *args, **kwargs) -> np.ndarray:
        pass
    
    @abstractmethod
    def train(self, 
              X: np.ndarray | torch.Tensor, 
              y:np.ndarray | torch.Tensor, *args, **kwargs) -> None:
        pass

class ConvolutionalNeuralNetwork(DigitClassificationInterface):
    def __init__(self, model) -> None:
        super().__init__()
        self.cnn = model

    def predict(self, image: np.ndarray | torch.Tensor, 
                *args, **kwargs) -> torch.Tensor:
        '''
        Takes a 28x28x1 image as input and provides a single integer value 
        that corresponds to a digit displayed on the image.
        
            Parameters:
                    image: An array representing and image of a digit.

            Returns:
                    prediction: A 0-9 digit as depicted on the image.
        '''
        test_output, last_layer = self.cnn(
            torch.tensor(image.reshape(-1, 1, 28, 28)))
        return torch.max(test_output, 1)[1].data.squeeze()
    
    def train(self, X, y, *args, **kwargs):
        raise NotImplementedError

class RandomForestClassifier(DigitClassificationInterface):
    def __init__(self, model) -> None:
        super().__init__()
        self.rf = model

    def predict(self, image: np.ndarray, *args, **kwargs) -> np.ndarray:
        '''
        Takes a 28x28x1 image as input and provides a single integer value 
        that corresponds to a digit displayed on the image.
        
            Parameters:
                    image: An array representing and image of a digit.

            Returns:
                    prediction: A 0-9 digit as depicted on the image.
        '''
        if len(image.shape) > 1:
            image = image.reshape(-1, 784)
        return np.array(self.rf.predict(image)[0], dtype=np.int32)
    
    def train(self, X, y, *args, **kwargs):
        raise NotImplementedError

class RandomModel(DigitClassificationInterface):
    def predict(self, image: np.ndarray) -> np.ndarray:
        '''
        Takes a 28x28x1 image as input and provides a single integer value 
        that corresponds to a digit displayed on the image.
        
            Parameters:
                    image: An array representing and image of a digit.

            Returns:
                    prediction: A 0-9 random number.
        '''
        image = image[9:-9, 9:-9]  # ?
        # Return a random digit between 0 and 9
        return random.randint(0, 9)
    
    def train(self, X, y, *args, **kwargs):
        raise NotImplementedError

class DigitClassifier:
    def __init__(self, algorithm: str) -> None:

        if algorithm == 'cnn':
            cnn = mnist.CNN()
            cnn.load_state_dict(torch.load('torch-model/cnn.torch'))
            cnn.eval()
            self.model = ConvolutionalNeuralNetwork(cnn)

        elif algorithm == 'rf':
            with open('rf.pickle', 'rb') as fr:
                rf = pickle.load(fr)
            self.model = RandomForestClassifier(rf)

        elif algorithm == 'rand':
            self.model = RandomModel()
        else:
            raise ValueError("Unsupported algorithm")

    def predict(self, image: np.ndarray | torch.Tensor) -> np.ndarray:
        '''
        Takes a 28x28x1 image as input and provides a single integer value 
        that corresponds to a digit displayed on the image.
        
            Parameters:
                    image: An array representing and image of a digit.

            Returns:
                    prediction: A 0-9 digit as depicted on the image,
                        or random number, depending on a selected classifier.
        '''
        return np.array(self.model.predict(image), dtype=np.int32)
    
    def train(self, X, y, *args, **kwargs):
        self.model.train(X, y, *args, **kwargs)

# Example usage:
# classifier = DigitClassifier('rand')
# prediction = classifier.predict(np.random.rand(28, 28, 1))
