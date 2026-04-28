# Model Search Leaderboard

- Train dataset: `data\raw\KDDTrain+.txt`
- External dataset: `data\raw\KDDTest+.txt`
- Best candidate: `baseline_small_weights`
- Best artifact directory: `artifacts\rare_class_search\best`

| Rank | Candidate | External Macro F1 | External U2R F1 | External R2L F1 | Validation Macro F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | baseline_small_weights | 0.5751 | 0.1239 | 0.3217 | 0.9169 |
| 2 | shallow_rare_tradeoff | 0.5690 | 0.1747 | 0.2741 | 0.9443 |
| 3 | baseline_prod | 0.5670 | 0.1345 | 0.2538 | 0.9274 |
| 4 | baseline_medium_weights | 0.5635 | 0.1333 | 0.3175 | 0.9219 |
| 5 | baseline_no_smote | 0.5160 | 0.1273 | 0.1052 | 0.9091 |

## Selected parameters

### baseline_small_weights

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

### shallow_rare_tradeoff

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

### baseline_medium_weights

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
    "r2l": 2.0,
    "u2r": 3.0
  }
}
```

### baseline_no_smote

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
  "use_smote": false
}
```
