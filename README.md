# Predictive Maintenance with KNN on Azure Machine Learning

**Student:** Rukha Manahil

## Problem

This project predicts machine failure from sensor readings using the AI4I 2020 Predictive Maintenance dataset from the UCI Machine Learning Repository (CC BY 4.0).

The dataset contains 10,000 machine records. Only about 3.4% of the records represent machine failures, making this an imbalanced binary classification problem.

## Approaches

Three Azure Machine Learning approaches were explored:
1. **Notebook (scikit-learn + MLflow)**  
   `notebooks/01_knn_notebook.ipynb`

2. **Automated ML (KNN only)**  
   `automl/automl_results.md`

3. **Designer (Execute Python Script)**  
   `designer/knn_designer_script.py`

## Results
| Metric | Notebook | Automated ML | Designer |
|---|---|---|---|
| Best K | 1 | 22 | 1 |
| Weights | uniform | distance | uniform |
| Scaler | StandardScaler | RobustScaler | StandardScaler |
| Test Recall (Failure) | 0.353 | N/A* | 0.353 |
| Test F1 (Failure) | 0.397 | N/A* | 0.397 |
| AUC | 0.669 | 0.919 (AUC weighted) | 0.669 |
| Code Written | Most | None / minimal SDK | Small script |
| Time to Set Up | Highest | Medium | Medium |
| Deployable to Managed Endpoint | Yes | Yes | No (classic components) |
| Best For | Full control and learning | Fast search and baselines | Visual teams and quick prototypes |

\*Azure AutoML reported macro/weighted classification metrics rather than the same failure-class recall and F1 used by the Notebook and Designer, so these values are not directly comparable.

The Automated ML experiment completed successfully. Its best model used **KNN with K=22, distance weighting, RobustScaler, and Manhattan distance**. The best model achieved **0.9714 accuracy** and **0.9186 weighted AUC**. Because AutoML's displayed macro/weighted metrics are calculated differently from the failure-class metrics used in the Notebook and Designer, they are not treated as direct replacements for failure recall and failure F1.

## Report Questions

### 1. Which approach gave the best recall on failures? Were the differences large?

The Notebook and Designer approaches both achieved a failure-class recall of approximately **0.353**, so their results were identical. Automated ML reported a **macro recall of 0.5862**, but this is an average across the classes and is not directly comparable to the failure-class recall reported by the Notebook and Designer. Therefore, a fair comparison of failure-class recall can only be made between the Notebook and Designer, where there was no difference.

### 2. Which approach would you choose for a real factory project, and why?

I would choose the **Notebook approach** because it provides the most control over preprocessing, scaling, hyperparameter tuning, evaluation, MLflow tracking, model registration, and deployment. This control is important in a real predictive-maintenance system where model behaviour and failure detection need to be carefully monitored.

### 3. Is 3.4% failures a problem for KNN? What could help?

Yes. The small percentage of failures creates a strong class imbalance, so most neighbouring observations are likely to belong to the non-failure class. This can make KNN miss real failures even when overall accuracy appears high. More failure examples, resampling techniques, threshold adjustment, or another classification algorithm that supports class weighting could improve failure detection.

## Deployment

The registered Notebook KNN model was deployed to an Azure Machine Learning managed online endpoint using:

- Deployment name: `blue`
- VM size: `Standard_DS2_v2`
- Authentication: Key-based
- Live traffic: 100% to `blue`

The deployed endpoint was successfully tested using both Azure ML Studio and a REST request.

REST test result:

- HTTP status: `200`
- Prediction: `[0, 1]`

Files:

- Sample request: `deployment/sample-request.json`
- REST test script: `deployment/test_endpoint.py`
- Deployment notebook: `notebooks/02_deploy_endpoint.ipynb`

The endpoint was deleted after testing to stop further charges.

## Screenshots

Project screenshots are stored in the `screenshots/` folder.

## What I Learned

I learned how to prepare and register data in Azure Machine Learning and train a KNN classification model using scikit-learn. I learned why feature scaling and class imbalance are important when using KNN. I also compared Notebook, Automated ML, and Designer workflows and used MLflow to register a trained model. Finally, I deployed the model as a managed online endpoint, tested it through a REST request, and safely deleted the endpoint after testing.
