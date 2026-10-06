# AutoML KNN Results

## Experiment
- Experiment name: `knn-predictive-maintenance-automl`
- Task: Classification
- Target column: `machine_failure`
- Primary metric: `AUC_weighted`
- Status: Completed

## Best Model
- Algorithm: KNeighborsClassifier (KNN)
- Scaler: RobustScaler
- Best K (n_neighbors): 22
- Weights: distance
- Distance metric: manhattan

## AutoML Metrics
- Accuracy: 0.9714000
- AUC weighted: 0.9186202
- AUC macro: 0.9186202
- AUC micro: 0.9935153
- F1 macro: 0.6374363
- F1 weighted: 0.9615796
- Balanced accuracy: 0.5862369
- Recall macro: 0.5862369
- Precision macro: 0.9495813

## Notes
Automated ML completed successfully using the numeric-encoded AI4I dataset.
The best AutoML model used RobustScaler with KNN, 22 neighbors,
distance weighting, and Manhattan distance.
