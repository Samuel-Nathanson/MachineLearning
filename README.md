# MachineLearning
Samuel E. Nathanson's repository of <i>Private Machine Learning Libraries</i>

## Installation
```
git lfs install
git clone --recurse-submodules https://github.com/Samuel-Nathanson/MachineLearning.git
```

## Repository Structure

### examples/
Within the _examples_ directory, one can find examples of working with each dataset.

### lib/
Contains library modules written by Samuel E. Nathanson

### data/
Pretty self-explanatory. 

### assignments/
You can find a copy of the assignments for this course here.

### submodules/
The submodules directory contains report LaTeX, pulled directly from Overleaf.

## Machine Learning Reports

### For the grader:
You can find the reports (5-Minute Video and Paper) in the _Reports_ directory.

### Pull Reports from Overleaf
This project contains a submodule, MachineLearningPapers, which can be updated with the following commands:
```
git fetch 
git pull
git submodule update --remote MachineLearningPapers
```

### Overleaf Link 
https://www.overleaf.com/project/6133864288d8a85f1dc0e0af

## Datasets - Description of Datasets
1. Breast Cancer [Classification]
Overview: This data describes characteristics of cell nuclei present in benign and malignant tumors.
Predictor: Diagnosis: M or B
Source: University of Wisconsin, 1993
URL: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29
2. Car Evaluation [Classification]
Overview: The data is on evaluations of car acceptability based on price, comfort, and technical
specifications.
Predictor: CAR: unacc, acc, good, vgood
Source: Jozef Stefan Institute, Yugoslavia (Slovenia), 1988
URL: https://archive.ics.uci.edu/ml/datasets/Car+Evaluation
3. Congressional Vote [Classification]
Overview: This data set includes votes for each of the U.S. House of Representatives Congressmen
on the 16 key votes identified by the Congressional Quarterly Almanac.
Predictor: Class: democrat, republican
Source: University of California, Irvine, 1987
URL: https://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records
Notes: Be careful with this data set since “?” does not indicate a missing attribute value. It actually
means “abstain.”
4. Abalone [Regression]
Overview: The data describes the physical measurements of abalone and the associated age.
Predictor: Rings (int)
Source: Marine Research Laboratories, Tasmania, 1995
URL: https://archive.ics.uci.edu/ml/datasets/Abalone
5. Computer Hardware [Regression]
Overview: The data describes relative CPU performance described by features such as cycle time,
memory size, etc.
Predictor: PRP (int)
Source: Tel Aviv University, Israel, 1987
URL: https://archive.ics.uci.edu/ml/datasets/Computer+Hardware
Notes: The estimated relative performance ERP values were estimated by the authors using a linear
regression method. This cannot be used as a feature. You should remove it from the feature
set, but save it elsewhere. In a later lab, you will have a chance to see how well you can replicate
the results with these two models ERP and PRP.
6. Forest Fires [Regression]
Overview: This is a difficult regression task, where the aim is to predict the burned area of forest
fires by using meteorological and other data.
Predictor: area (float)
Source: University of Minho, Portugal, 2007
URL: https://archive.ics.uci.edu/ml/datasets/Forest+Fires
Notes: The output area is very skewed toward 0.0. The authors recommend a log transform.
