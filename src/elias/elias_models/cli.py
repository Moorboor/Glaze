from __future__ import annotations

import argparse
import sys

import pandas as pd

from .data_loading import load_participant_data, preprocess_loaded_participant_data
from .reporting import (
    Step5PipelineError,
    build_step5_pipeline_config,
    run_step345_pipeline,
)
from .surrogate_recovery import (
    build_step3_pipeline_config,
    list_step3_runs,
    load_step3_run,
    run_step3_pipeline,
)
from .train_test_eval import (
    build_step4_pipeline_config,
    list_step4_runs,
    load_step4_run,
    run_step4_pipeline,
)


def _parse_candidate_models(raw_value: str) -> tuple[str, ...]:
    models = tuple(part.strip() for part in str(raw_value).split(",") if part.strip())
    if not models:
        raise ValueError("candidate model list is empty")
    return models


def _parse_participant_ids(raw_value: str | None) -> list[str] | None:
    if raw_value is None:
        return None
    ids = [part.strip() for part in str(raw_value).split(",") if part.strip()]
    return ids if ids else None


def _load_preprocessed_dataset(
    csv_path: str,
    hazard_col: str,
    participant_ids: list[str] | None = None,
) -> pd.DataFrame:
    loaded_df = load_participant_data(
        csv_path=csv_path,
        participant_ids=participant_ids,
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

    participant_run_parser = subparsers.add_parser(
        "participant-run",
        help="Run Step 4 participant train/test pipeline and persist outputs.",
    )
    participant_run_parser.add_argument("--run-id", type=str, required=True)
    participant_run_parser.add_argument("--output-root", type=str, default="data/elias")
    participant_run_parser.add_argument("--csv-path", type=str, default="data/participants.csv")
    participant_run_parser.add_argument("--hazard-col", type=str, default="subjective_h_snapshot")
    participant_run_parser.add_argument(
        "--participant-ids",
        type=str,
        default=None,
        help="Optional comma-separated participant IDs.",
    )
    participant_run_parser.add_argument(
        "--candidate-models",
        type=str,
        default="cont_threshold,cont_asymptote,ddm_dnm",
        help="Comma-separated candidate model names.",
    )
    participant_run_parser.add_argument("--fit-n-starts", type=int, default=4)
    participant_run_parser.add_argument("--fit-n-iterations", type=int, default=8)
    participant_run_parser.add_argument("--fit-n-sims-per-trial", type=int, default=150)
    participant_run_parser.add_argument("--eval-n-sims-per-trial", type=int, default=150)
    participant_run_parser.add_argument("--rt-bin-width-ms", type=float, default=20.0)
    participant_run_parser.add_argument("--rt-max-ms", type=float, default=5000.0)
    participant_run_parser.add_argument("--eps", type=float, default=1e-12)
    participant_run_parser.add_argument("--dt-ms", type=float, default=1.0)
    participant_run_parser.add_argument("--max-duration-ms", type=float, default=5000.0)
    participant_run_parser.add_argument(
        "--winner-primary-score-column",
        type=str,
        default="joint_score",
        choices=["joint_score", "choice_only_score", "rt_only_cond_score", "bic_score"],
    )
    participant_run_parser.add_argument("--winner-tie-tolerance", type=float, default=1e-9)
    participant_run_parser.add_argument("--seed", type=int, default=0)
    participant_run_parser.add_argument("--overwrite", action="store_true")

    participant_show_parser = subparsers.add_parser(
        "participant-show",
        help="Show summary for one persisted Step 4 run.",
    )
    participant_show_parser.add_argument("--run-id", type=str, required=True)
    participant_show_parser.add_argument("--output-root", type=str, default="data/elias")

    participant_list_parser = subparsers.add_parser(
        "participant-list",
        help="List persisted Step 4 runs.",
    )
    participant_list_parser.add_argument("--output-root", type=str, default="data/elias")

    pipeline_parser = subparsers.add_parser(
        "pipeline-run",
        help="Run combined Step 3, Step 4, and Step 5 pipelines with linked manifests.",
    )
    pipeline_parser.add_argument("--run-id", type=str, required=True)
    pipeline_parser.add_argument("--output-root", type=str, default="data/elias")
    pipeline_parser.add_argument("--csv-path", type=str, default="data/participants.csv")
    pipeline_parser.add_argument("--hazard-col", type=str, default="subjective_h_snapshot")
    pipeline_parser.add_argument(
        "--participant-ids",
        type=str,
        default=None,
        help="Optional comma-separated participant IDs.",
    )
    pipeline_parser.add_argument(
        "--candidate-models",
        type=str,
        default="cont_threshold,cont_asymptote,ddm_dnm",
        help="Comma-separated candidate model names used by both steps.",
    )
    pipeline_parser.add_argument("--seed", type=int, default=0)
    pipeline_parser.add_argument("--dt-ms", type=float, default=1.0)
    pipeline_parser.add_argument("--max-duration-ms", type=float, default=5000.0)
    pipeline_parser.add_argument("--overwrite", action="store_true")

    pipeline_parser.add_argument("--step3-n-surrogates-per-model", type=int, default=20)
    pipeline_parser.add_argument("--step3-surrogate-n-draws-per-trial", type=int, default=128)
    pipeline_parser.add_argument("--step3-fit-n-starts", type=int, default=4)
    pipeline_parser.add_argument("--step3-fit-n-iterations", type=int, default=8)
    pipeline_parser.add_argument("--step3-fit-n-sims-per-trial", type=int, default=150)
    pipeline_parser.add_argument("--step3-soft-gate-joint-diag-min", type=float, default=0.60)
    pipeline_parser.add_argument("--step3-soft-gate-param-median-r-min", type=float, default=0.30)

    pipeline_parser.add_argument("--step4-fit-n-starts", type=int, default=4)
    pipeline_parser.add_argument("--step4-fit-n-iterations", type=int, default=8)
    pipeline_parser.add_argument("--step4-fit-n-sims-per-trial", type=int, default=150)
    pipeline_parser.add_argument("--step4-eval-n-sims-per-trial", type=int, default=150)
    pipeline_parser.add_argument("--step4-rt-bin-width-ms", type=float, default=20.0)
    pipeline_parser.add_argument("--step4-rt-max-ms", type=float, default=5000.0)
    pipeline_parser.add_argument("--step4-eps", type=float, default=1e-12)
    pipeline_parser.add_argument(
        "--step4-winner-primary-score-column",
        type=str,
        default="joint_score",
        choices=["joint_score", "choice_only_score", "rt_only_cond_score", "bic_score"],
    )
    pipeline_parser.add_argument("--step4-winner-tie-tolerance", type=float, default=1e-9)
    pipeline_parser.add_argument("--step5-ppc-n-sims-per-trial", type=int, default=200)
    pipeline_parser.add_argument("--step5-ddm-n-samples-per-trial", type=int, default=200)
    pipeline_parser.add_argument("--step5-rt-bin-width-ms", type=float, default=20.0)
    pipeline_parser.add_argument("--step5-rt-max-ms", type=float, default=5000.0)
    pipeline_parser.add_argument("--step5-eps", type=float, default=1e-12)
    pipeline_parser.add_argument(
        "--step5-seed",
        type=int,
        default=None,
        help="Optional override seed for Step 5. Defaults to --seed when omitted.",
    )
    pipeline_parser.add_argument("--step5-latent-cont-noise-std", type=float, default=0.0)

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


def _cmd_participant_run(args: argparse.Namespace) -> None:
    candidate_models = _parse_candidate_models(args.candidate_models)
    participant_ids = _parse_participant_ids(args.participant_ids)
    config = build_step4_pipeline_config(
        candidate_models=candidate_models,
        fit_n_starts=int(args.fit_n_starts),
        fit_n_iterations=int(args.fit_n_iterations),
        fit_n_sims_per_trial=int(args.fit_n_sims_per_trial),
        eval_n_sims_per_trial=int(args.eval_n_sims_per_trial),
        rt_bin_width_ms=float(args.rt_bin_width_ms),
        rt_max_ms=float(args.rt_max_ms),
        eps=float(args.eps),
        dt_ms=float(args.dt_ms),
        max_duration_ms=float(args.max_duration_ms),
        random_seed=int(args.seed),
        winner_primary_score_column=str(args.winner_primary_score_column),
        winner_tie_tolerance=float(args.winner_tie_tolerance),
    )

    df_all = _load_preprocessed_dataset(
        args.csv_path,
        args.hazard_col,
        participant_ids=participant_ids,
    )
    run_output = run_step4_pipeline(
        df_all,
        run_id=str(args.run_id),
        output_root=str(args.output_root),
        config=config,
        overwrite=bool(args.overwrite),
    )

    manifest = run_output["manifest"]
    print(f"Run finished: {manifest['run_id']}")
    print(f"Run directory: {run_output['run_dir']}")
    print(f"Participants: {manifest['n_participants']} | Fit rows: {manifest['n_fit_rows']}")
    print(f"Group winner model: {manifest['group_winner_model_name']}")


def _cmd_participant_show(args: argparse.Namespace) -> None:
    loaded = load_step4_run(run_id=str(args.run_id), output_root=str(args.output_root))
    manifest = loaded["manifest"]
    tables = loaded["tables"]

    print(f"Run: {manifest.get('run_id')}")
    print(f"Created: {manifest.get('created_at_utc')}")
    print(f"Status: {manifest.get('status')}")
    print(f"Run directory: {loaded['run_dir']}")

    if "group_winner_summary" in tables and not tables["group_winner_summary"].empty:
        print("\nGroup winner summary:")
        print(tables["group_winner_summary"].to_string(index=False))

    if "participant_winner_table" in tables and not tables["participant_winner_table"].empty:
        print("\nParticipant winners:")
        print(tables["participant_winner_table"].to_string(index=False))

    if "participant_model_scores_test" in tables and not tables["participant_model_scores_test"].empty:
        summary = (
            tables["participant_model_scores_test"]
            .groupby(["participant_id", "candidate_model_name"], as_index=False)
            .agg(
                joint_score=("joint_score", "first"),
                bic_score=("bic_score", "first"),
            )
            .sort_values(["participant_id", "joint_score", "candidate_model_name"])
            .reset_index(drop=True)
        )
        print("\nParticipant-model TEST scores:")
        print(summary.to_string(index=False))


def _cmd_participant_list(args: argparse.Namespace) -> None:
    runs = list_step4_runs(output_root=str(args.output_root))
    if runs.empty:
        print("No Step 4 runs found.")
        return
    print(runs.to_string(index=False))


def _cmd_pipeline_run(args: argparse.Namespace) -> None:
    candidate_models = _parse_candidate_models(args.candidate_models)
    participant_ids = _parse_participant_ids(args.participant_ids)

    step3_config = build_step3_pipeline_config(
        candidate_models=candidate_models,
        n_surrogates_per_model=int(args.step3_n_surrogates_per_model),
        surrogate_n_draws_per_trial=int(args.step3_surrogate_n_draws_per_trial),
        fit_n_starts=int(args.step3_fit_n_starts),
        fit_n_iterations=int(args.step3_fit_n_iterations),
        fit_n_sims_per_trial=int(args.step3_fit_n_sims_per_trial),
        dt_ms=float(args.dt_ms),
        max_duration_ms=float(args.max_duration_ms),
        random_seed=int(args.seed),
        soft_gate_joint_diag_min=float(args.step3_soft_gate_joint_diag_min),
        soft_gate_param_median_r_min=float(args.step3_soft_gate_param_median_r_min),
    )
    step4_config = build_step4_pipeline_config(
        candidate_models=candidate_models,
        fit_n_starts=int(args.step4_fit_n_starts),
        fit_n_iterations=int(args.step4_fit_n_iterations),
        fit_n_sims_per_trial=int(args.step4_fit_n_sims_per_trial),
        eval_n_sims_per_trial=int(args.step4_eval_n_sims_per_trial),
        rt_bin_width_ms=float(args.step4_rt_bin_width_ms),
        rt_max_ms=float(args.step4_rt_max_ms),
        eps=float(args.step4_eps),
        dt_ms=float(args.dt_ms),
        max_duration_ms=float(args.max_duration_ms),
        random_seed=int(args.seed),
        winner_primary_score_column=str(args.step4_winner_primary_score_column),
        winner_tie_tolerance=float(args.step4_winner_tie_tolerance),
    )
    step5_seed = int(args.step5_seed) if args.step5_seed is not None else int(args.seed)
    step5_config = build_step5_pipeline_config(
        ppc_n_sims_per_trial=int(args.step5_ppc_n_sims_per_trial),
        ddm_n_samples_per_trial=int(args.step5_ddm_n_samples_per_trial),
        rt_bin_width_ms=float(args.step5_rt_bin_width_ms),
        rt_max_ms=float(args.step5_rt_max_ms),
        eps=float(args.step5_eps),
        random_seed=int(step5_seed),
        latent_cont_noise_std=float(args.step5_latent_cont_noise_std),
    )

    df_all = _load_preprocessed_dataset(
        args.csv_path,
        args.hazard_col,
        participant_ids=participant_ids,
    )
    try:
        pipeline_output = run_step345_pipeline(
            df_all,
            run_id=str(args.run_id),
            output_root=str(args.output_root),
            step3_config=step3_config,
            step4_config=step4_config,
            step5_config=step5_config,
            overwrite=bool(args.overwrite),
        )
    except Step5PipelineError as exc:
        print(f"Master run failed at Step 5: {args.run_id}", file=sys.stderr)
        if exc.manifest_path is not None:
            print(f"Master manifest path: {exc.manifest_path}", file=sys.stderr)
        if exc.error_log_path is not None:
            print(f"Step 5 error log path: {exc.error_log_path}", file=sys.stderr)
        raise

    print(f"Master run finished: {pipeline_output['run_id']}")
    print(f"Master manifest path: {pipeline_output['manifest_path']}")
    print(f"Linked Step 3 run: {pipeline_output['step3_run_id']}")
    print(f"Linked Step 4 run: {pipeline_output['step4_run_id']}")
    print(f"Step 5 status: {pipeline_output['manifest']['step5_status']}")
    print(f"Step 5 report path: {pipeline_output['step5_report_path']}")


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
    if args.command == "participant-run":
        _cmd_participant_run(args)
        return
    if args.command == "participant-show":
        _cmd_participant_show(args)
        return
    if args.command == "participant-list":
        _cmd_participant_list(args)
        return
    if args.command == "pipeline-run":
        _cmd_pipeline_run(args)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
