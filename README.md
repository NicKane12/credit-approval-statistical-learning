# Predicting Credit Approval Using Statistical Learning

## Overview

This project analyzes credit approval data to find a classifcation model that can predict approval with high accuracy. The goal is compare and contrast various models using AUC and accuracy.

## Methods
- Logistic
- LDA
- QDA
- KNN 
- Classification Tree
- Random Forests
- Boosting
- Ridge
- LASSO

## Evaluation
Models were evaluated using accuracy (correct classification) and AUC using the ROC curve.

## Key Findings
- KNN and rnadom forests provided the highest level of accuracy
- Logistic regression, although having the lowest accuracy, had the highest AUC along with random forests.
- Random forests balances both as it has a sweet spot accuracy/AUC and allows for nonlinearity in model. 

## Data
The dataset consists of credit approval data from Green's Econometric Analysis textbook. The raw data can be found in the `data/` folder.

## Repo Structure
- `analysis/`: R script containing EDA and modeling.
- `report/`: final project report and slides.
- `data/`: data used on the project.

## How to Run
1. Open the script in `analysis/` using RStudio  
2. Install required packages listed at the top of the file  
3. Run the script sequentially to reproduce analysis and forecasts
