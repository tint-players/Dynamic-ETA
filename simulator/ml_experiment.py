from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch
from torch import nn

from .ml_baselines import (
    CurrentSpeedETABaseline,
    EffectiveCeilingETABaseline,
    ETAMetrics,
    HistoricalMeanSpeedETABaseline,
    eta_metrics_from_predictions,
    evaluate_eta_predictor,
)
from .ml_dataset import MLDatasetSplit, MLTrainingDatasetBuilder, split_training_samples_by_run
from .ml_features import CONTEXT_FEATURE_NAMES, GRAPH_FEATURE_NAMES, SEQUENCE_FEATURE_NAMES, SEQUENCE_WINDOW_STEPS
from .ml_gnn import GraphSAGEETARegressor, collate_gnn_training_samples, train_gnn_epoch
from .ml_hybrid import HybridLSTMGraphSAGEETARegressor, collate_hybrid_training_samples, train_hybrid_epoch
from .ml_lstm import LSTMETARegressor, collate_lstm_training_samples, train_lstm_epoch
from .ml_samples import MLTrainingSample
from .scenario_generator import GeneratedScenarioRun


@dataclass(frozen=True, slots=True)
class ETAExperimentConfig:
    """Reproducible training/evaluation settings for the model ablation ladder."""

    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    split_seed: int = 42
    training_seed: int = 42
    early_stopping_patience: int = 4
    min_validation_improvement_s: float = 0.5
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    sample_every_n_steps: int = 1
    lstm_hidden_size: int = 128
    lstm_layers: int = 2
    gnn_hidden_size: int = 128
    gnn_layers: int = 3
    dropout: float = 0.10
    device: str | None = None

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive")
        if self.min_validation_improvement_s < 0:
            raise ValueError("min_validation_improvement_s cannot be negative")
        if min(self.train_ratio, self.validation_ratio, self.test_ratio) < 0:
            raise ValueError("split ratios cannot be negative")
        if self.sample_every_n_steps <= 0:
            raise ValueError("sample_every_n_steps must be positive")
        if self.lstm_hidden_size <= 0 or self.lstm_layers <= 0:
            raise ValueError("LSTM dimensions must be positive")
        if self.gnn_hidden_size <= 0 or self.gnn_layers <= 0:
            raise ValueError("GNN dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True, slots=True)
class CandidateExperimentResult:
    name: str
    validation_metrics: ETAMetrics
    training_losses: tuple[float, ...] = ()
    best_epoch: int | None = None


@dataclass(slots=True)
class ETAExperimentResult:
    config: ETAExperimentConfig
    split: MLDatasetSplit
    candidates: tuple[CandidateExperimentResult, ...]
    winner_name: str
    winner_validation_metrics: ETAMetrics
    winner_test_metrics: ETAMetrics
    winner_state_dict: dict[str, torch.Tensor] | None
    winner_model_kwargs: dict[str, object]

    def candidate(self, name: str) -> CandidateExperimentResult:
        return next(item for item in self.candidates if item.name == name)


@dataclass(slots=True)
class _TrainedCandidate:
    result: CandidateExperimentResult
    model: nn.Module
    model_kwargs: dict[str, object]


def build_experiment_samples(
    runs: Iterable[GeneratedScenarioRun],
    *,
    sample_every_n_steps: int = 1,
) -> tuple[MLTrainingSample, ...]:
    """Build samples with each run's exact randomized scenario config."""

    if sample_every_n_steps <= 0:
        raise ValueError("sample_every_n_steps must be positive")
    materialized = list(runs)
    if not materialized:
        raise ValueError("At least one generated run is required")

    seen_run_ids: set[str] = set()
    samples: list[MLTrainingSample] = []
    node_ids: tuple[str, ...] | None = None
    edge_index: tuple[tuple[int, ...], tuple[int, ...]] | None = None

    for generated_run in materialized:
        if generated_run.run_id in seen_run_ids:
            raise ValueError(f"Duplicate generated run_id: {generated_run.run_id}")
        seen_run_ids.add(generated_run.run_id)
        run_samples = MLTrainingDatasetBuilder(generated_run.config).build_run_samples(
            generated_run.frames,
            sample_every_n_steps=sample_every_n_steps,
        )
        if not run_samples:
            raise ValueError(f"Generated run {generated_run.run_id} produced no ML samples")

        first_inputs = run_samples[0].inputs
        if node_ids is None:
            node_ids = first_inputs.node_ids
            edge_index = first_inputs.edge_index
        elif first_inputs.node_ids != node_ids or first_inputs.edge_index != edge_index:
            raise ValueError("All experiment runs must share one canonical graph topology")
        samples.extend(run_samples)

    return tuple(samples)


def _chunks(samples: Sequence[MLTrainingSample], batch_size: int):
    for start in range(0, len(samples), batch_size):
        yield samples[start:start + batch_size]


def _evaluate_lstm_batched(
    model: LSTMETARegressor,
    samples: Sequence[MLTrainingSample],
    *,
    batch_size: int,
    device: torch.device,
) -> ETAMetrics:
    model.eval()
    predictions: list[float] = []
    targets: list[float] = []
    with torch.no_grad():
        for chunk in _chunks(samples, batch_size):
            batch = collate_lstm_training_samples(chunk, device=device)
            predictions.extend(float(value) for value in model(batch.x_seq, batch.x_context).cpu())
            targets.extend(float(value) for value in batch.y.cpu())
    return eta_metrics_from_predictions(predictions, targets)


def _evaluate_gnn_batched(
    model: GraphSAGEETARegressor,
    samples: Sequence[MLTrainingSample],
    *,
    batch_size: int,
    device: torch.device,
) -> ETAMetrics:
    model.eval()
    predictions: list[float] = []
    targets: list[float] = []
    with torch.no_grad():
        for chunk in _chunks(samples, batch_size):
            batch = collate_gnn_training_samples(chunk, device=device)
            prediction = model(
                batch.x_graph,
                batch.edge_index,
                batch.current_node_index,
                batch.route_mask,
                batch.x_context,
            )
            predictions.extend(float(value) for value in prediction.cpu())
            targets.extend(float(value) for value in batch.y.cpu())
    return eta_metrics_from_predictions(predictions, targets)


def _evaluate_hybrid_batched(
    model: HybridLSTMGraphSAGEETARegressor,
    samples: Sequence[MLTrainingSample],
    *,
    batch_size: int,
    device: torch.device,
) -> ETAMetrics:
    model.eval()
    predictions: list[float] = []
    targets: list[float] = []
    with torch.no_grad():
        for chunk in _chunks(samples, batch_size):
            batch = collate_hybrid_training_samples(chunk, device=device)
            prediction = model(
                batch.x_seq,
                batch.x_graph,
                batch.edge_index,
                batch.current_node_index,
                batch.route_mask,
                batch.x_context,
            )
            predictions.extend(float(value) for value in prediction.cpu())
            targets.extend(float(value) for value in batch.y.cpu())
    return eta_metrics_from_predictions(predictions, targets)


def _train_neural_candidate(
    *,
    name: str,
    model: nn.Module,
    model_kwargs: dict[str, object],
    train_samples: Sequence[MLTrainingSample],
    validation_samples: Sequence[MLTrainingSample],
    experiment: ETAExperimentConfig,
    train_epoch: Callable[..., float],
    evaluate: Callable[..., ETAMetrics],
    device: torch.device,
) -> _TrainedCandidate:
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=experiment.learning_rate,
        weight_decay=experiment.weight_decay,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: ETAMetrics | None = None
    best_epoch: int | None = None
    stale_epochs = 0
    training_losses: list[float] = []

    for epoch in range(1, experiment.epochs + 1):
        loss = train_epoch(
            model,
            train_samples,
            optimizer,
            batch_size=experiment.batch_size,
            device=device,
            shuffle_seed=experiment.training_seed + epoch,
        )
        training_losses.append(float(loss))
        metrics = evaluate(
            model,
            validation_samples,
            batch_size=experiment.batch_size,
            device=device,
        )

        improved = (
            best_metrics is None
            or metrics.mae_s < best_metrics.mae_s - experiment.min_validation_improvement_s
        )
        if improved:
            best_metrics = metrics
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= experiment.early_stopping_patience:
                break

    if best_state is None or best_metrics is None or best_epoch is None:
        raise RuntimeError(f"{name} training did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    return _TrainedCandidate(
        result=CandidateExperimentResult(
            name=name,
            validation_metrics=best_metrics,
            training_losses=tuple(training_losses),
            best_epoch=best_epoch,
        ),
        model=model,
        model_kwargs=model_kwargs,
    )


def run_eta_experiment(
    runs: Iterable[GeneratedScenarioRun],
    *,
    experiment: ETAExperimentConfig | None = None,
) -> ETAExperimentResult:
    """Train the complete ETA ablation ladder on one frozen run-level split.

    Model selection uses validation MAE only. The held-out test partition is
    evaluated exactly once, after the winner has been selected, so it cannot
    influence architecture or checkpoint selection.
    """

    experiment = experiment or ETAExperimentConfig()
    materialized_runs = tuple(runs)
    if len(materialized_runs) < 3:
        raise ValueError("At least three runs are required for train/validation/test experimentation")

    samples = build_experiment_samples(
        materialized_runs,
        sample_every_n_steps=experiment.sample_every_n_steps,
    )
    split = split_training_samples_by_run(
        samples,
        train_ratio=experiment.train_ratio,
        validation_ratio=experiment.validation_ratio,
        test_ratio=experiment.test_ratio,
        split_seed=experiment.split_seed,
    )
    if not split.train or not split.validation or not split.test:
        raise ValueError("Experiment split must contain train, validation, and test samples")

    route_lengths = {round(run.config.route.total_length_m, 6) for run in materialized_runs}
    if len(route_lengths) != 1:
        raise ValueError("Current baseline comparison requires one shared route length")
    base_config = materialized_runs[0].config

    candidates: list[CandidateExperimentResult] = []
    baseline_models = {
        "current_speed": CurrentSpeedETABaseline.from_config(base_config),
        "historical_mean_speed": HistoricalMeanSpeedETABaseline.from_config(base_config),
        "effective_ceiling": EffectiveCeilingETABaseline.from_config(base_config),
    }
    for name, baseline in baseline_models.items():
        candidates.append(
            CandidateExperimentResult(
                name=name,
                validation_metrics=evaluate_eta_predictor(baseline, split.validation),
            )
        )

    device = torch.device(
        experiment.device
        if experiment.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(experiment.training_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(experiment.training_seed)

    lstm_kwargs: dict[str, object] = {
        "hidden_size": experiment.lstm_hidden_size,
        "num_layers": experiment.lstm_layers,
        "dropout": experiment.dropout,
    }
    lstm = _train_neural_candidate(
        name="lstm",
        model=LSTMETARegressor(**lstm_kwargs),
        model_kwargs=lstm_kwargs,
        train_samples=split.train,
        validation_samples=split.validation,
        experiment=experiment,
        train_epoch=train_lstm_epoch,
        evaluate=_evaluate_lstm_batched,
        device=device,
    )
    candidates.append(lstm.result)

    gnn_kwargs: dict[str, object] = {
        "hidden_size": experiment.gnn_hidden_size,
        "num_layers": experiment.gnn_layers,
        "dropout": experiment.dropout,
    }
    gnn = _train_neural_candidate(
        name="graphsage",
        model=GraphSAGEETARegressor(**gnn_kwargs),
        model_kwargs=gnn_kwargs,
        train_samples=split.train,
        validation_samples=split.validation,
        experiment=experiment,
        train_epoch=train_gnn_epoch,
        evaluate=_evaluate_gnn_batched,
        device=device,
    )
    candidates.append(gnn.result)

    hybrid_kwargs: dict[str, object] = {
        "lstm_hidden_size": experiment.lstm_hidden_size,
        "lstm_layers": experiment.lstm_layers,
        "gnn_hidden_size": experiment.gnn_hidden_size,
        "gnn_layers": experiment.gnn_layers,
        "dropout": experiment.dropout,
    }
    hybrid = _train_neural_candidate(
        name="hybrid",
        model=HybridLSTMGraphSAGEETARegressor(**hybrid_kwargs),
        model_kwargs=hybrid_kwargs,
        train_samples=split.train,
        validation_samples=split.validation,
        experiment=experiment,
        train_epoch=train_hybrid_epoch,
        evaluate=_evaluate_hybrid_batched,
        device=device,
    )
    candidates.append(hybrid.result)

    winner = min(candidates, key=lambda item: (item.validation_metrics.mae_s, item.name))
    trained = {"lstm": lstm, "graphsage": gnn, "hybrid": hybrid}

    if winner.name in baseline_models:
        winner_test_metrics = evaluate_eta_predictor(baseline_models[winner.name], split.test)
        winner_state_dict = None
        winner_model_kwargs = {
            "route_length_m": float(base_config.route.total_length_m),
            "speed_floor_kmh": float(baseline_models[winner.name].speed_floor_kmh),
        }
    else:
        trained_winner = trained[winner.name]
        if winner.name == "lstm":
            winner_test_metrics = _evaluate_lstm_batched(
                trained_winner.model,
                split.test,
                batch_size=experiment.batch_size,
                device=device,
            )
        elif winner.name == "graphsage":
            winner_test_metrics = _evaluate_gnn_batched(
                trained_winner.model,
                split.test,
                batch_size=experiment.batch_size,
                device=device,
            )
        else:
            winner_test_metrics = _evaluate_hybrid_batched(
                trained_winner.model,
                split.test,
                batch_size=experiment.batch_size,
                device=device,
            )
        winner_state_dict = {
            key: value.detach().cpu().clone()
            for key, value in trained_winner.model.state_dict().items()
        }
        winner_model_kwargs = copy.deepcopy(trained_winner.model_kwargs)

    return ETAExperimentResult(
        config=experiment,
        split=split,
        candidates=tuple(candidates),
        winner_name=winner.name,
        winner_validation_metrics=winner.validation_metrics,
        winner_test_metrics=winner_test_metrics,
        winner_state_dict=winner_state_dict,
        winner_model_kwargs=winner_model_kwargs,
    )


def save_winner_checkpoint(result: ETAExperimentResult, path: str | Path) -> None:
    """Persist the selected model plus the feature/split contract needed to audit it."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model_name": result.winner_name,
        "model_kwargs": result.winner_model_kwargs,
        "model_state_dict": result.winner_state_dict,
        "experiment_config": asdict(result.config),
        "validation_metrics": asdict(result.winner_validation_metrics),
        "test_metrics": asdict(result.winner_test_metrics),
        "split_run_ids": {
            "train": result.split.train_run_ids,
            "validation": result.split.validation_run_ids,
            "test": result.split.test_run_ids,
        },
        "feature_contract": {
            "sequence_window_steps": SEQUENCE_WINDOW_STEPS,
            "sequence_feature_names": SEQUENCE_FEATURE_NAMES,
            "graph_feature_names": GRAPH_FEATURE_NAMES,
            "context_feature_names": CONTEXT_FEATURE_NAMES,
        },
    }
    torch.save(payload, destination)
