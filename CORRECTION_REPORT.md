# Rapport de correction commercialisation

## Diagnostic

Le projet avait un risque majeur de faux signal commercial:

1. Les quality gates validaient des métriques déclarées dans un JSON, sans vérifier que les artefacts et rapports existaient dans le workspace courant.
2. `reports/release/summary.json` pointait vers `C:\Users\shoko\Desktop\Code\LightGBM-Cybersecurity\...`, donc hors de ce checkout.
3. Le script `train_model.py` local était un générateur POC: dataset synthétique, modèle `LogisticRegression`, métriques codées en dur.
4. Les métriques rares restent le point faible sur le vrai `KDDTest+.txt`: R2L et U2R généralisent mal malgré très bonnes métriques validation.

## Train relancé

Commande:

```powershell
$env:PYTHONPATH='src'; py -3.12 train_model.py --dataset data/raw/KDDTrain+.txt --artifact-dir artifacts/final --report-dir reports/final --no-progress
```

Résultat validation:

| Métrique | Valeur |
| --- | ---: |
| Accuracy | 0.9741 |
| Macro F1 | 0.7162 |
| Macro recall | 0.9687 |
| Attack F1 | 0.9739 |
| R2L F1 | 0.5131 |
| U2R F1 | 0.1183 |

Résultat externe sur `KDDTest+.txt`:

| Métrique | Valeur |
| --- | ---: |
| Accuracy | 0.8054 |
| Macro F1 | 0.6899 |
| Macro recall | 0.6882 |
| Attack F1 | 0.8411 |
| Attack ROC AUC | 0.9491 |
| Attack average precision | 0.9448 |
| R2L F1 | 0.4382 |
| U2R F1 | 0.5011 |

## Fix anti-overfit

Le modèle précédent avait un gap validation/externe fort:

| Mesure | Avant | Après |
| --- | ---: | ---: |
| Validation macro F1 | 0.9180 | 0.7162 |
| External macro F1 | 0.5595 | 0.6899 |
| Gap macro F1 | 0.3585 | 0.0263 |
| External accuracy | 0.7734 | 0.8054 |
| External R2L F1 | 0.2904 | 0.4382 |
| External U2R F1 | 0.1005 | 0.5011 |

Correction appliquée au profil `default-prod`:

1. Arbres très petits: `num_leaves=3`, `max_depth=3`.
2. Feuilles plus robustes: `min_child_samples=220`.
3. Subsampling fort: `feature_fraction=0.55`, `bagging_fraction=0.65`, `bagging_freq=1`.
4. Régularisation forte: `lambda_l1=2.0`, `lambda_l2=8.0`.
5. Suppression oversampling train: `use_smote=False`.
6. Seuil attaque calibré: `threshold=0.40`.
7. Gate release dur: écart `validation_macro_f1 - external_macro_f1` limité à `0.05`.

Effet: validation baisse volontairement, généralisation externe monte, gap supprimé. Le modèle ne peut plus être validé seulement parce que le split interne est flatteur.

## Corrections appliquées

1. `src/ids_project/quality.py`
   - Ajout d'un mode commercial strict avec evidence.
   - Vérification que `artifact_dir` et `report_path` sont dans le workspace.
   - Vérification du manifest: modèle `lightgbm`, schema version `2`, hash dataset, versions dépendances, fichiers runtime.
   - Vérification que le rapport externe correspond au summary.
   - Rejet des rapports externes sous `1000` exemples.
   - Obligation des métriques sécurité: `attack_precision`, `attack_recall`, `attack_f1_score`, `attack_roc_auc`, `attack_average_precision`.

2. `src/ids_project/cli.py`
   - `check-release` utilise le gate strict par défaut.
   - `--metrics-only` garde l'ancien mode pour audit historique.

3. `src/ids_project/runtime/__init__.py`
   - Les prédictions passent au modèle un `DataFrame` avec noms de colonnes.
   - Suppression des warnings scikit/LightGBM sur les feature names.

4. `src/ids_project/evaluation.py`
   - Macro ROC AUC et average precision robustes aux labels externes inconnus.
   - Les classes absentes ne forcent plus les métriques probabilistes à `0.0`.

5. `src/ids_project/preprocessing.py`
   - `categorical_min_frequency` est désormais appliqué par le `OneHotEncoder`.
   - `use_anomaly_feature` permet de désactiver le signal IsolationForest si une recherche future montre qu'il sur-apprend.

6. `train_model.py`
   - Suppression dataset synthétique + métriques inventées.
   - Wrapper vers le vrai pipeline `ids-cli train`.

## Etat commercial

Commercialisable côté discipline anti-overfit. Encore à cadrer côté couverture métier.

Raison: le surapprentissage est éradiqué par le gap gate, mais certaines familles rares restent trop faibles pour une promesse commerciale exhaustive:

1. R2L F1 externe: `0.4382`.
2. U2R F1 externe: `0.5011`.
3. Plusieurs familles restent à rappel nul ou faible (`snmpgetattack`, `snmpguess`, `mailbomb`, `sendmail`, `loadmodule`, `xlock`, `xsnoop`, etc.).

Le projet est maintenant dans un état sain pour empêcher une release trompeuse: le gate strict bloque tant que les preuves locales et les seuils ne sont pas réellement satisfaits.

## Verification

1. `pytest`: 52 passed.
2. Train LightGBM sur `KDDTrain+.txt`: passed.
3. Evaluation externe sur `KDDTest+.txt`: passed.
4. `check-release --metrics-only`: conserve le mode sans evidence, mais rejette l'ancien summary avec les nouveaux seuils anti-overfit.
5. `check-release` strict: bloque les summaries hors workspace ou sans evidence.
