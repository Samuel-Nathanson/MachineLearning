# CVT to Jupyter

from BreastCancer import preprocessBreastCancer
from lib.KNN import kNNCondenseTrainingSet, predict, plotDataSet

numFolds = 5
folds = preprocessBreastCancer(numFolds)

yColumnId = "class"

prediction = predict(3, folds[0], folds[1].drop(columns=yColumnId).iloc[1], yColumnId)
print(prediction)

k = 3
plotDataSet(k, folds[0], yColumnId)

# print(kNNCondenseTrainingSet(folds[0], "class"))