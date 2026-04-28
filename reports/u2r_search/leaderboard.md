# Model Search Leaderboard

- Train dataset: `data\raw\KDDTrain+.txt`
- External dataset: `data\raw\KDDTest+.txt`
- Best candidate: `u2r_weight_2_5`
- Best artifact directory: `artifacts\u2r_search\best`

| Rank | Candidate | External Macro F1 | External U2R F1 | External R2L F1 | Validation Macro F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | u2r_weight_2_5 | 0.5756 | 0.1239 | 0.3343 | 0.9218 |
| 2 | u2r_weight_2_5_regularized | 0.5753 | 0.1504 | 0.3201 | 0.9401 |
| 3 | rare_class_prod | 0.5751 | 0.1239 | 0.3217 | 0.9169 |
| 4 | shallow_u2r_weighted | 0.5736 | 0.1674 | 0.3235 | 0.9451 |
| 5 | u2r_weight_3 | 0.5729 | 0.1228 | 0.3273 | 0.9283 |

## Selected parameters

### u2r_weight_2_5

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
  "use_smote": true,
  "custom_class_weights": {
    "normal": 1.0,
    "dos": 1.0,
    "probe": 1.0,
    "r2l": 1.5,
    "u2r": 2.5
  }
}
```

### u2r_weight_2_5_regularized

```json
{
  "n_estimators": 220,
  "learning_rate": 0.05,
  "num_leaves": 31,
  "min_child_samples": 30,
  "feature_fraction": 0.95,
  "bagging_fraction": 0.9,
  "bagging_freq": 1,
  "lambda_l1": 0.1,
  "lambda_l2": 0.5,
  "use_smote": true,
  "custom_class_weights": {
    "normal": 1.0,
    "dos": 1.0,
    "probe": 1.0,
    "r2l": 1.5,
    "u2r": 2.5
  }
}
```

### rare_class_prod

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
  "use_smote": true,
  "custom_class_weights": {
    "normal": 1.0,
    "dos": 1.0,
    "probe": 1.0,
    "r2l": 1.5,
    "u2r": 2.0
  }
}
```

### shallow_u2r_weighted

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
  "use_smote": true,
  "custom_class_weights": {
    "normal": 1.0,
    "dos": 1.0,
    "probe": 1.0,
    "r2l": 1.5,
    "u2r": 2.5
  }
}
```

### u2r_weight_3

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
  "use_smote": true,
  "custom_class_weights": {
    "normal": 1.0,
    "dos": 1.0,
    "probe": 1.0,
    "r2l": 1.5,
    "u2r": 3.0
  }
}
```
