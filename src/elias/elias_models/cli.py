from __future__ import annotations

import argparse

from .data_loading import load_participant_data
from .orchestration import run_all_models_for_participant


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for the smoke-test demo."""
    parser = argparse.ArgumentParser(description="Run participant-wise model simulations.")
    parser.add_argument(
        "--csv-path",
        type=str,
        default="data/participants.csv",
        help="Path to participant CSV file.",
    )
    parser.add_argument(
        "--participant-id",
        type=str,
        default="P01",
        help="Participant id to simulate in demo mode.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="DDM Monte Carlo samples per trial.",
    )
    parser.add_argument(
        "--include-rt-samples",
        action="store_true",
        help="Include raw DDM RT sample arrays in output.",
    )
    parser.add_argument(
        "--hazard-col",
        type=str,
        default="subjective_h_snapshot",
        help="Column to use as hazard input H.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic simulations.",
    )
    return parser


def main() -> None:
    """Run CLI entrypoint for participant-wise model simulation smoke tests."""
    args = _build_arg_parser().parse_args()

    loaded_df = load_participant_data(
        csv_path=args.csv_path,
        participant_ids=[args.participant_id],
        hazard_col=args.hazard_col,
        reset_on=("participant", "block"),
    )

    outputs = run_all_models_for_participant(
        loaded_df,
        participant_id=args.participant_id,
        random_seed=args.seed,
        n_samples_per_trial=args.samples,
        include_rt_samples=args.include_rt_samples,
    )

    print(
        f"Loaded {len(loaded_df)} rows for participant {args.participant_id}. "
        f"Running models: {', '.join(outputs.keys())}"
    )

    for model_name, model_df in outputs.items():
        mean_rt = (
            float(model_df["predicted_rt_ms"].mean()) if not model_df.empty else float("nan")
        )
        timeout_rate = (
            float((model_df["predicted_decision"] == 0).mean())
            if not model_df.empty
            else float("nan")
        )
        print(
            f"[{model_name}] rows={len(model_df)} "
            f"mean_predicted_rt_ms={mean_rt:.2f} "
            f"timeout_rate={timeout_rate:.3f}"
        )


if __name__ == "__main__":
    main()
