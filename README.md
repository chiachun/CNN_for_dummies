# CNN for dummies
A simplified CNN.

## dependency
pandas numpy

## layers
input -> conv(one 3x3 filter) -> relu -> maxpool(3x3, stride=3) -> softmax

## test dataset
from Kaggle(https://www.kaggle.com/c/digit-recognizer/data)

## result
training samples: the first 100 entries of 0 and 1

test samples: the last 1000 entires of 0 and 1

accuracy: 99.5%  