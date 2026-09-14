from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from simulator.config_loader import load_simulation_config
from simulator.ml_experiment import ETAExperimentConfig, run_eta_experiment, save_winner_checkpoint
from simulator.scenario_generator import ScenarioGenerator, ScenarioGeneratorConfig


def _metrics_dict(metrics):
    return asdict(metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare ETA baselines, LSTM, GraphSAGE, and hybrid model")
    parser.add_argument("--scenario", default="examples/delhi_agra_corridor.yaml")
    parser.add_argument("--runs", type=int, default=30, help="number of randomized simulation runs")
    parser.add_argument("--scenario-seed", type=int, default=20260914)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--lstm-hidden-size", type=int, default=128)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--gnn-hidden-size", type=int, default=128)
    parser.add_argument("--gnn-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--device", default=None, help="torch device, e.g. cpu, cuda, cuda:0")
    parser.add_argument("--checkpoint", default="artifacts/eta_winner.pt")
    parser.add_argument("--summary", default="artifacts/eta_experiment_summary.json")
    args = parser.parse_args()

    if args.runs < 3:
        raise SystemExit("--runs must be at least 3 so train/validation/test are all represented")

    base_config = load_simulation_config(args.scenario)
    generator = ScenarioGenerator(
        base_config,
        ScenarioGeneratorConfig(
            n_scenarios=args.runs,
            random_seed=args.scenario_seed,
        ),
    )
    runs = generator.generate_runs()

    experiment_config = ETAExperimentConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        split_seed=args.split_seed,
        training_seed=args.training_seed,
        early_stopping_patience=args.patience,
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_layers=args.lstm_layers,
        gnn_hidden_size=args.gnn_hidden_size,
        gnn_layers=args.gnn_layers,
        dropout=args.dropout,
        device=args.device,
    )
    result = run_eta_experiment(runs, experiment=experiment_config)
    save_winner_checkpoint(result, args.checkpoint)

    summary = {
        "generated_runs": len(runs),
        "scenario_seed": generator.random_seed,
        "sample_count": result.split.sample_count,
        "split": {
            "train_runs": result.split.train_run_ids,
            "validation_runs": result.split.validation_run_ids,
            "test_runs": result.split.test_run_ids,
            "train_samples": len(result.split.train),
            "validation_samples": len(result.split.validation),
            "test_samples": len(result.split.test),
        },
        "validation": {
            candidate.name: {
                "metrics": _metrics_dict(candidate.validation_metrics),
                "training_losses": candidate.training_losses,
                "best_epoch": candidate.best_epoch,
            }
            for candidate in result.candidates
        },
        "winner": result.winner_name,
        "winner_validation_metrics": _metrics_dict(result.winner_validation_metrics),
        "winner_test_metrics": _metrics_dict(result.winner_test_metrics),
        "checkpoint": str(Path(args.checkpoint)),
    }

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
