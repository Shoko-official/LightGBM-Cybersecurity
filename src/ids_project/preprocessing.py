from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, StandardScaler

from ids_project.config import TrainingConfig
from ids_project.contracts import (
    CATEGORICAL_COLUMNS,
    NSL_KDD_CATEGORY_MAP,
    NSL_KDD_COLUMNS,
    NUMERIC_COLUMNS,
    SKEWED_COLUMNS,
)

REMAINING_NUMERIC_COLUMNS = [column for column in NUMERIC_COLUMNS if column not in SKEWED_COLUMNS]


class NumericClipper(BaseEstimator, TransformerMixin):
    def __init__(self, quantile: float = 0.995):
        self.quantile = quantile
        self.lower_bounds_: np.ndarray | None = None
        self.upper_bounds_: np.ndarray | None = None

    def fit(self, frame, target=None) -> "NumericClipper":
        values = np.asarray(frame, dtype=float)
        self.lower_bounds_ = np.quantile(values, 1 - self.quantile, axis=0)
        self.upper_bounds_ = np.quantile(values, self.quantile, axis=0)
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, frame):
        if self.lower_bounds_ is None or self.upper_bounds_ is None:
            raise ValueError("NumericClipper must be fitted before transform.")
        values = np.asarray(frame, dtype=float)
        return np.clip(values, self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self.to_drop_: list[int] = []

    def fit(self, frame, target=None) -> "CorrelationFilter":
        matrix = np.asarray(frame, dtype=float)
        self.n_features_in_ = matrix.shape[1]
        if matrix.shape[1] <= 1:
            return self
        with np.errstate(invalid="ignore", divide="ignore"):
            corr_matrix = np.abs(np.nan_to_num(np.corrcoef(matrix, rowvar=False), nan=0.0))
        self.to_drop_ = []
        for left in range(corr_matrix.shape[1]):
            for right in range(left + 1, corr_matrix.shape[1]):
                if corr_matrix[left, right] > self.threshold and right not in self.to_drop_:
                    self.to_drop_.append(right)
        self.to_drop_ = sorted(set(self.to_drop_))
        return self

    def transform(self, frame):
        matrix = np.asarray(frame, dtype=float)
        return np.delete(matrix, self.to_drop_, axis=1)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        return np.delete(np.asarray(input_features, dtype=object), self.to_drop_)


class AnomalyFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, contamination: float = 0.05):
        self.contamination = contamination
        self.model = IsolationForest(contamination=contamination, n_jobs=1, random_state=42)

    def fit(self, frame, target=None) -> "AnomalyFeatureExtractor":
        values = np.asarray(frame, dtype=float)
        self.n_features_in_ = values.shape[1]
        self.model.fit(values)
        return self

    def transform(self, frame):
        scores = self.model.decision_function(frame).reshape(-1, 1)
        return np.hstack([frame, scores])

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        return np.append(np.asarray(input_features, dtype=object), "unsupervised_anomaly_score")


@dataclass(slots=True)
class IDSLabelEncoder:
    benign_label: str = "normal"
    category_map: dict[str, str] = field(default_factory=lambda: NSL_KDD_CATEGORY_MAP.copy())
    classes_: list[str] = field(default_factory=list)

    def fit(self, labels: pd.Series) -> "IDSLabelEncoder":
        categories = labels.astype(str).str.strip().str.lower().map(self.category_map).fillna("unknown")
        ordered = [self.benign_label, "dos", "probe", "r2l", "u2r", "unknown"]
        self.classes_ = [label for label in ordered if label in set(categories)]
        return self

    def transform(self, labels: pd.Series) -> pd.Series:
        categories = labels.astype(str).str.strip().str.lower().map(self.category_map).fillna("unknown")
        return categories.apply(lambda value: self.classes_.index(value) if value in self.classes_ else -1)

    def fit_transform(self, labels: pd.Series) -> pd.Series:
        self.fit(labels)
        return self.transform(labels)

    @property
    def mapping(self) -> dict[str, int]:
        return {label: index for index, label in enumerate(self.classes_)}

    def inverse_transform(self, indices: np.ndarray) -> list[str]:
        return [self.classes_[index] if 0 <= index < len(self.classes_) else "unknown" for index in indices]


@dataclass(slots=True)
class PreprocessingArtifacts:
    pipeline: Pipeline
    feature_names: list[str]
    label_encoder: IDSLabelEncoder


def build_preprocessor(config: TrainingConfig) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clipper", NumericClipper(quantile=config.numeric_clip_quantile)),
            ("scaler", StandardScaler()),
        ]
    )
    skewed_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("clipper", NumericClipper(quantile=config.numeric_clip_quantile)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )

    return Pipeline(
        steps=[
            (
                "union",
                ColumnTransformer(
                    transformers=[
                        ("skewed", skewed_pipeline, SKEWED_COLUMNS),
                        ("numeric", numeric_pipeline, REMAINING_NUMERIC_COLUMNS),
                        ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
                    ]
                ),
            ),
            ("correlation_filter", CorrelationFilter(threshold=0.98)),
            ("selector", VarianceThreshold(threshold=0.0)),
            ("anomaly_extractor", AnomalyFeatureExtractor(contamination=0.05)),
        ]
    )


def fit_preprocessing(
    frame: pd.DataFrame,
    labels: pd.Series,
    config: TrainingConfig,
) -> tuple[np.ndarray, pd.Series, PreprocessingArtifacts]:
    _validate_feature_frame(frame)
    label_encoder = IDSLabelEncoder(benign_label=config.positive_label)
    encoded_labels = label_encoder.fit_transform(labels)
    pipeline = build_preprocessor(config)
    transformed = pipeline.fit_transform(frame)
    if hasattr(pipeline, "get_feature_names_out"):
        feature_names = pipeline.get_feature_names_out().tolist()
    else:
        feature_names = [f"f{index}" for index in range(transformed.shape[1])]

    return transformed, encoded_labels, PreprocessingArtifacts(
        pipeline=pipeline,
        feature_names=feature_names,
        label_encoder=label_encoder,
    )


def transform_features(frame: pd.DataFrame, artifacts: PreprocessingArtifacts) -> np.ndarray:
    _validate_feature_frame(frame)
    return artifacts.pipeline.transform(frame)


def transform_labels(labels: pd.Series, artifacts: PreprocessingArtifacts) -> pd.Series:
    return artifacts.label_encoder.transform(labels)


def _validate_feature_frame(frame: pd.DataFrame) -> None:
    expected_columns = set(NSL_KDD_COLUMNS)
    frame_columns = set(frame.columns)
    missing = sorted(expected_columns - frame_columns)
    extra = sorted(frame_columns - expected_columns)
    if missing:
        raise ValueError(f"Input features are missing columns: {', '.join(missing)}")
    if extra:
        raise ValueError(f"Input features contain unexpected columns: {', '.join(extra)}")
