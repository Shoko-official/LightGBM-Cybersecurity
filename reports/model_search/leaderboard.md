# Model Search Leaderboard

- Train dataset: `data\raw\KDDTrain+.txt`
- External dataset: `data\raw\KDDTest+.txt`
- Best candidate: `baseline_prod`
- Best artifact directory: `artifacts\model_search\best`

| Rank | Candidate | External Macro F1 | External U2R F1 | External R2L F1 | Validation Macro F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | baseline_prod | 0.5708 | 0.1435 | 0.2989 | 0.9327 |
| 2 | shallow_regularized | 0.5690 | 0.1747 | 0.2741 | 0.9443 |
| 3 | rare_class_focus | 0.5689 | 0.1429 | 0.3013 | 0.9387 |
| 4 | regularized_bagging | 0.5678 | 0.1600 | 0.2610 | 0.9334 |
| 5 | wider_trees | 0.5551 | 0.1504 | 0.2532 | 0.9333 |

## Selected parameters

### baseline_prod

```json
{
  "n_estimators": 200,
  "learning_rate": 0.05,
  "num_leaves": 31,
  "min_child_samples": 20,
  "feature_fraction": 1.0,
  "bagging_fraction": 1.0,
  "bagging_freq": 0,
  "lambda_l1": 0.0,
  "lambda_l2": 0.0,
  "use_smote": true
}
```

### shallow_regularized

```json
{
  "n_estimators": 260,
  "learning_rate": 0.06,
  "num_leaves": 15,
  "max_depth": 8,
  "min_child_samples": 40,
  "feature_fraction": 0.85,
  "bagging_fraction": 0.8,
  "bagging_freq": 1,
  "lambda_l1": 0.2,
  "lambda_l2": 1.0,
  "use_smote": true
}
```

### rare_class_focus

```json
{
  "n_estimators": 240,
  "learning_rate": 0.03,
  "num_leaves": 63,
  "min_child_samples": 10,
  "feature_fraction": 0.95,
  "bagging_fraction": 0.9,
  "bagging_freq": 1,
  "lambda_l1": 0.0,
  "lambda_l2": 0.2,
  "use_smote": true
}
```

### regularized_bagging

```json
{
  "n_estimators": 240,
  "learning_rate": 0.04,
  "num_leaves": 31,
  "min_child_samples": 40,
  "feature_fraction": 0.9,
  "bagging_fraction": 0.8,
  "bagging_freq": 1,
  "lambda_l1": 0.0,
  "lambda_l2": 1.0,
  "use_smote": true
}
```

### wider_trees

```json
{
  "n_estimators": 220,
  "learning_rate": 0.04,
  "num_leaves": 63,
  "min_child_samples": 20,
  "feature_fraction": 0.9,
  "bagging_fraction": 0.9,
  "bagging_freq": 1,
  "lambda_l1": 0.0,
  "lambda_l2": 0.5,
  "use_smote": true
}
```
