from __future__ import annotations

import argparse

import pandas as pd

from .data_loading import load_participant_data, preprocess_loaded_participant_data
from .surrogate_recovery import (
    build_step3_pipeline_config,
    list_step3_runs,
    load_step3_run,
    run_step3_pipeline,
)


def _parse_candidate_models(raw_value: str) -> tuple[str, ...]:
    models = tuple(part.strip() for part in str(raw_value).split(",") if part.strip())
    if not models:
        raise ValueError("candidate model list is empty")
    return models


def _load_preprocessed_dataset(csv_path: str, hazard_col: str) -> pd.DataFrame:
    loaded_df = load_participant_data(
        csv_path=csv_path,
        participant_ids=None,
        hazard_col=hazard_col,
        reset_on=("participant", "block"),
    )
    prep = preprocess_loaded_participant_data(loaded_df)
    return prep["df_all"]


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for persistent Elias pipelines."""
    parser = argparse.ArgumentParser(description="Elias model pipelines")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "surrogate-run",
        help="Run Step 3 surrogate-recovery pipeline and persist outputs.",
    )
    run_parser.add_argument("--run-id", type=str, required=True)
    run_parser.add_argument("--output-root", type=str, default="data/elias")
    run_parser.add_argument("--csv-path", type=str, default="data/participants.csv")
    run_parser.add_argument("--hazard-col", type=str, default="subjective_h_snapshot")
    run_parser.add_argument(
        "--candidate-models",
        type=str,
        default="cont_threshold,cont_asymptote,ddm_dnm",
        help="Comma-separated candidate model names.",
    )
    run_parser.add_argument("--n-surrogates-per-model", type=int, default=20)
    run_parser.add_argument("--surrogate-n-draws-per-trial", type=int, default=128)
    run_parser.add_argument("--fit-n-starts", type=int, default=4)
    run_parser.add_argument("--fit-n-iterations", type=int, default=8)
    run_parser.add_argument("--fit-n-sims-per-trial", type=int, default=150)
    run_parser.add_argument("--dt-ms", type=float, default=1.0)
    run_parser.add_argument("--max-duration-ms", type=float, default=5000.0)
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument("--overwrite", action="store_true")

    show_parser = subparsers.add_parser(
        "surrogate-show",
        help="Show summary for one persisted Step 3 run.",
    )
    show_parser.add_argument("--run-id", type=str, required=True)
    show_parser.add_argument("--output-root", type=str, default="data/elias")

    list_parser = subparsers.add_parser(
        "surrogate-list",
        help="List persisted Step 3 runs.",
    )
    list_parser.add_argument("--output-root", type=str, default="data/elias")

    return parser


def _cmd_surrogate_run(args: argparse.Namespace) -> None:
    candidate_models = _parse_candidate_models(args.candidate_models)
    config = build_step3_pipeline_config(
        candidate_models=candidate_models,
        n_surrogates_per_model=int(args.n_surrogates_per_model),
        surrogate_n_draws_per_trial=int(args.surrogate_n_draws_per_trial),
        fit_n_starts=int(args.fit_n_starts),
        fit_n_iterations=int(args.fit_n_iterations),
        fit_n_sims_per_trial=int(args.fit_n_sims_per_trial),
        dt_ms=float(args.dt_ms),
        max_duration_ms=float(args.max_duration_ms),
        random_seed=int(args.seed),
    )

    df_all = _load_preprocessed_dataset(args.csv_path, args.hazard_col)

    run_output = run_step3_pipeline(
        df_all,
        run_id=str(args.run_id),
        output_root=str(args.output_root),
        config=config,
        overwrite=bool(args.overwrite),
    )

    manifest = run_output["manifest"]
    print(f"Run finished: {manifest['run_id']}")
    print(f"Run directory: {run_output['run_dir']}")
    print(f"Soft-gate status: {manifest['soft_gate']['overall_status']}")
    print(f"Surrogates: {manifest['n_surrogates_total']} | Fit rows: {manifest['n_fit_rows']}")


def _cmd_surrogate_show(args: argparse.Namespace) -> None:
    loaded = load_step3_run(run_id=str(args.run_id), output_root=str(args.output_root))
    manifest = loaded["manifest"]
    tables = loaded["tables"]

    print(f"Run: {manifest.get('run_id')}")
    print(f"Created: {manifest.get('created_at_utc')}")
    print(f"Status: {manifest.get('status')}")
    print(f"Run directory: {loaded['run_dir']}")

    if "soft_gate_summary" in tables and not tables["soft_gate_summary"].empty:
        print("\nSoft-gate summary:")
        print(tables["soft_gate_summary"].to_string(index=False))

    if "fit_results" in tables and not tables["fit_results"].empty:
        summary = (
            tables["fit_results"]
            .groupby(["generating_model_name", "candidate_model_name"], as_index=False)
            .agg(
                n_rows=("joint_score", "size"),
                joint_min=("joint_score", "min"),
                joint_median=("joint_score", "median"),
            )
            .sort_values(["generating_model_name", "joint_median", "candidate_model_name"])
            .reset_index(drop=True)
        )
        print("\nFit summary:")
        print(summary.to_string(index=False))


def _cmd_surrogate_list(args: argparse.Namespace) -> None:
    runs = list_step3_runs(output_root=str(args.output_root))
    if runs.empty:
        print("No Step 3 runs found.")
        return
    print(runs.to_string(index=False))


def main() -> None:
    """Run CLI entrypoint for Elias model pipelines."""
    args = _build_arg_parser().parse_args()

    if args.command == "surrogate-run":
        _cmd_surrogate_run(args)
        return
    if args.command == "surrogate-show":
        _cmd_surrogate_show(args)
        return
    if args.command == "surrogate-list":
        _cmd_surrogate_list(args)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
