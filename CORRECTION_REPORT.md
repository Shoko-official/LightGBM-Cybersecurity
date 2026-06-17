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
| Accuracy | 0.9989 |
| Macro F1 | 0.9180 |
| Macro recall | 0.9234 |
| Attack F1 | 0.9989 |
| R2L F1 | 0.9848 |
| U2R F1 | 0.6087 |

Résultat externe sur `KDDTest+.txt`:

| Métrique | Valeur |
| --- | ---: |
| Accuracy | 0.7734 |
| Macro F1 | 0.5595 |
| Macro recall | 0.5333 |
| Attack F1 | 0.7772 |
| Attack ROC AUC | 0.9532 |
| Attack average precision | 0.9491 |
| R2L F1 | 0.2904 |
| U2R F1 | 0.1005 |

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

Non commercialisable en l'état.

Raison: le modèle entraîné est réel, mais la performance externe rare-class reste sous seuil commercial raisonnable:

1. R2L F1 externe: `0.2904`.
2. U2R F1 externe: `0.1005`.
3. Plusieurs familles R2L/U2R ont rappel nul ou très faible (`guess_passwd`, `snmpgetattack`, `httptunnel`, `loadmodule`, `perl`, `rootkit`, etc.).

Le projet est maintenant dans un état sain pour empêcher une release trompeuse: le gate strict bloque tant que les preuves locales et les seuils ne sont pas réellement satisfaits.

## Verification

1. `pytest`: 52 passed.
2. Train LightGBM sur `KDDTrain+.txt`: passed.
3. Evaluation externe sur `KDDTest+.txt`: passed.
4. `check-release --metrics-only`: garde compat historique.
5. `check-release` strict: bloque les summaries hors workspace ou sans evidence.
