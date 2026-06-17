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
| Accuracy | 0.9965 |
| Macro F1 | 0.8803 |
| Macro recall | 0.9609 |
| Attack F1 | 0.9964 |
| R2L F1 | 0.9120 |
| U2R F1 | 0.5000 |

Résultat externe sur `KDDTest+.txt`:

| Métrique | Valeur |
| --- | ---: |
| Accuracy | 0.8074 |
| Macro F1 | 0.6209 |
| Macro recall | 0.5892 |
| Attack F1 | 0.8352 |
| Attack ROC AUC | 0.9581 |
| Attack average precision | 0.9591 |
| R2L F1 | 0.3957 |
| U2R F1 | 0.2205 |

## Fix anti-overfit

Le modèle précédent avait un gap validation/externe fort:

| Mesure | Avant | Après |
| --- | ---: | ---: |
| Validation macro F1 | 0.9180 | 0.8803 |
| External macro F1 | 0.5595 | 0.6209 |
| Gap macro F1 | 0.3585 | 0.2595 |
| External accuracy | 0.7734 | 0.8074 |
| External R2L F1 | 0.2904 | 0.3957 |
| External U2R F1 | 0.1005 | 0.2205 |

Correction appliquée au profil `default-prod`:

1. Arbres plus petits: `num_leaves=7`, `max_depth=5`.
2. Feuilles plus robustes: `min_child_samples=80`.
3. Subsampling: `feature_fraction=0.75`, `bagging_fraction=0.75`, `bagging_freq=1`.
4. Régularisation: `lambda_l1=0.8`, `lambda_l2=3.0`.
5. Suppression oversampling train: `use_smote=False`.
6. Seuil attaque plus sensible: `threshold=0.30`.

Effet: validation baisse volontairement, généralisation externe monte. C'est le comportement attendu quand on réduit le surapprentissage.

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

5. `train_model.py`
   - Suppression dataset synthétique + métriques inventées.
   - Wrapper vers le vrai pipeline `ids-cli train`.

## Etat commercial

Pas encore totalement commercialisable.

Raison: le surapprentissage est réduit, mais certaines familles rares restent trop faibles pour une promesse commerciale forte:

1. R2L F1 externe: `0.3957`.
2. U2R F1 externe: `0.2205`.
3. Plusieurs familles restent à rappel nul ou faible (`snmpgetattack`, `snmpguess`, `mailbomb`, `sendmail`, `loadmodule`, `xlock`, `xsnoop`, etc.).

Le projet est maintenant dans un état sain pour empêcher une release trompeuse: le gate strict bloque tant que les preuves locales et les seuils ne sont pas réellement satisfaits.

## Verification

1. `pytest`: 52 passed.
2. Train LightGBM sur `KDDTrain+.txt`: passed.
3. Evaluation externe sur `KDDTest+.txt`: passed.
4. `check-release --metrics-only`: conserve le mode sans evidence, mais rejette l'ancien summary avec les nouveaux seuils anti-overfit.
5. `check-release` strict: bloque les summaries hors workspace ou sans evidence.
