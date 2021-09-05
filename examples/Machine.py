from lib.PreprocessingTK import *
import pandas
import numpy as np
data = pandas.read_csv("../data/Machine/machine.data",
                  names=["VendorName", "ModelName", "MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP", "ERP"])

vendorNames = np.unique(data["VendorName"])
modelNames = np.unique(data["ModelName"])

convertToNominal(data, "VendorName", vendorNames, inplace=True)
convertToNominal(data, "ModelName", modelNames, inplace=True)
print("One-hot Coded Nominal Data for VendorName: " + str(data["VendorName"][0]))

frequencyDiscretizedData = discretize(data, "PRP", xargs={"dMethod": "frequency", "bins": 2})
zeroIndex = np.min(frequencyDiscretizedData["PRP"])
oneIndex = np.max(frequencyDiscretizedData["PRP"])
print("Frequency Discretized - PRP Bin 0: " + str(np.count_nonzero(frequencyDiscretizedData["PRP"] == zeroIndex)))
print("Frequency Discretized - PRP Bin 1: " + str(np.count_nonzero(frequencyDiscretizedData["PRP"] == oneIndex)))

discretize(data, "PRP", xargs={"dMethod": "equal-width", "bins": 2}, inplace=True)
zeroIndex = np.min(data["PRP"])
oneIndex = np.max(data["PRP"])
print("Equal-Width Discretized - PRP Bin 0: " + str(np.count_nonzero(data["PRP"] == zeroIndex)))
print("Equal-Width Discretized - PRP Bin 1: " + str(np.count_nonzero(data["PRP"] == oneIndex)))

# TODO: Test that this works
# Use function to split training / testing set before standardizing
standardize(data, data, "MMAX", inplace=True)

partition(data, 5, classificationColumnId=None, includeValidationSet=True)

print("MSE Error: " + str(evaluateError([1,2,3,4,5], [2,3,4,5,6], method="MSE")))
evaluateError([1,2,3,4,5], [2,3,4,5,6], method="MAE")
evaluateError([0,0,0,1,0], [1,1,0,1,0], method="precision")
evaluateError([0,0,0,1,0], [1,1,0,1,0], method="recall")



