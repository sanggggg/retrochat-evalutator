"""CLI entry points for Retrochat Evaluator."""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import click
from dotenv import load_dotenv
from langchain_core.callbacks import get_usage_metadata_callback

from .config import Config, TrainingConfig, EvaluationConfig, SummarizationMethod
from .models.rubric import RubricList, Rubric
from .models.chat_session import ChatSession
from .training.trainer import Trainer
from .training.llm_summarizer import RubricSummarizer
from .training.semantic_summarizer import SemanticClusteringSummarizer
from .training.visualizer import save_clustering_visualization
from .evaluation.evaluator import Evaluator
from .validation.validator import Validator
from .llm.gemini import GeminiClient


def setup_logging(verbose: bool) -> None:
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def get_model_pricing(model_name: str) -> tuple[float, float]:
    """Get input and output token pricing per 1M tokens for a model.

    Args:
        model_name: The model name (e.g., 'gemini-2.5-pro', 'gemini-2.5-flash-lite')

    Returns:
        Tuple of (input_price_per_1M, output_price_per_1M) in USD.
        Returns (0.0, 0.0) if model pricing is not found.

    Reference: https://ai.google.dev/gemini-api/docs/pricing
    """
    model_lower = model_name.lower()

    # Gemini 3 Pro Preview
    if "gemini-3-pro" in model_lower and "image" not in model_lower:
        # For prompts <= 200k tokens: $1.00 input, $6.00 output
        # For prompts > 200k tokens: $2.00 input, $9.00 output
        # We'll use the <= 200k pricing as default (most common case)
        return (1.00, 6.00)

    # Gemini 2.5 Pro
    if "gemini-2.5-pro" in model_lower and "flash" not in model_lower:
        # For prompts <= 200k tokens: $0.625 input, $1.25 output
        # For prompts > 200k tokens: $5.00 input, $7.50 output
        return (0.625, 5.00)

    # Gemini 2.5 Flash
    if "gemini-2.5-flash" in model_lower and "lite" not in model_lower:
        return (0.15, 1.25)

    # Gemini 2.5 Flash-Lite
    if "gemini-2.5-flash-lite" in model_lower:
        return (0.05, 0.20)

    # Default: unknown model, return 0
    return (0.0, 0.0)


def calculate_cost(
    input_tokens: int, output_tokens: int, input_price: float, output_price: float
) -> float:
    """Calculate total cost in USD.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        input_price: Price per 1M input tokens
        output_price: Price per 1M output tokens

    Returns:
        Total cost in USD
    """
    input_cost = (input_tokens / 1_000_000) * input_price
    output_cost = (output_tokens / 1_000_000) * output_price
    return input_cost + output_cost


def format_token_usage(usage_metadata: dict) -> None:
    """Format and display token usage with cost information.

    Args:
        usage_metadata: Token usage metadata from LangChain callback
    """
    if not usage_metadata:
        return

    click.echo("\n" + "=" * 60)
    click.echo("Token Usage Summary:")
    click.echo("=" * 60)

    total_input = 0
    total_output = 0
    total_tokens = 0
    total_cost = 0.0

    for model_name, usage in usage_metadata.items():
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        total = usage.get("total_tokens", 0)

        total_input += input_tokens
        total_output += output_tokens
        total_tokens += total

        # Get pricing for this model
        input_price, output_price = get_model_pricing(model_name)
        cost = calculate_cost(input_tokens, output_tokens, input_price, output_price)
        total_cost += cost

        click.echo(f"\n{model_name}:")
        click.echo(f"  Input tokens:  {input_tokens:,}")
        click.echo(f"  Output tokens: {output_tokens:,}")
        click.echo(f"  Total tokens:  {total:,}")

        if input_price > 0 or output_price > 0:
            input_cost = (input_tokens / 1_000_000) * input_price
            output_cost = (output_tokens / 1_000_000) * output_price
            click.echo(f"  Input cost:    ${input_cost:.6f} (${input_price:.3f} per 1M tokens)")
            click.echo(f"  Output cost:   ${output_cost:.6f} (${output_price:.3f} per 1M tokens)")
            click.echo(f"  Total cost:    ${cost:.6f}")
        else:
            click.echo(f"  Cost:          Unknown pricing for this model")

    click.echo("\n" + "-" * 60)
    click.echo("Total across all models:")
    click.echo(f"  Input tokens:  {total_input:,}")
    click.echo(f"  Output tokens: {total_output:,}")
    click.echo(f"  Total tokens:  {total_tokens:,}")
    if total_cost > 0:
        click.echo(f"  Total cost:    ${total_cost:.6f} USD")
    click.echo("=" * 60)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """Retrochat Evaluator - AI Agent Efficiency Evaluation using LLM-as-a-judge."""
    load_dotenv()
    setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


@main.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to YAML config file (overrides other options)",
)
@click.option(
    "--dataset-dir",
    "-d",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Directory containing JSONL session files",
)
@click.option(
    "--dataset-manifest",
    "-m",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to dataset manifest JSON file",
)
@click.option(
    "--score-top-percentile",
    type=click.FloatRange(0.0, 100.0),
    default=None,
    help="Keep only the top N percentile of sessions by score before training",
)
@click.option(
    "--score-name",
    "-n",
    type=str,
    default=None,
    help="Name of the score to use for filtering (default: 'default')",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for training results (default: ./output)",
)
@click.option(
    "--prompts-dir",
    "-p",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Directory containing prompt templates (default: prompts/)",
)
@click.option(
    "--summarization-method",
    type=click.Choice(["llm", "semantic_clustering"], case_sensitive=False),
    default=None,
    help="Method for consolidating rubrics: 'llm' (prompt-based) or 'semantic_clustering' (embedding + HAC)",
)
@click.option(
    "--embedding-model",
    type=str,
    default=None,
    help="Google AI embedding model for semantic clustering (default: models/text-embedding-004)",
)
@click.option(
    "--umap-n-neighbors",
    type=int,
    default=None,
    help="UMAP n_neighbors for semantic clustering (default: 15)",
)
@click.option(
    "--umap-n-components",
    type=int,
    default=None,
    help="UMAP n_components for semantic clustering (default: 5)",
)
@click.option(
    "--min-cluster-size",
    type=int,
    default=None,
    help="HDBSCAN min_cluster_size for semantic clustering (default: 2)",
)
@click.option(
    "--min-rubrics",
    type=int,
    default=None,
    help="Minimum number of rubrics to output (default: 5)",
)
@click.option(
    "--max-rubrics",
    type=int,
    default=None,
    help="Maximum number of rubrics to output (default: 10)",
)
@click.pass_context
def train(
    ctx: click.Context,
    config: Path | None,
    dataset_dir: Path | None,
    dataset_manifest: Path | None,
    score_top_percentile: float | None,
    score_name: str | None,
    output: Path | None,
    prompts_dir: Path | None,
    summarization_method: str | None,
    embedding_model: str | None,
    umap_n_neighbors: int | None,
    umap_n_components: int | None,
    min_cluster_size: int | None,
    min_rubrics: int | None,
    max_rubrics: int | None,
) -> None:
    """Train evaluation rubrics from high-quality chat sessions."""
    # Load config from YAML if provided
    if config:
        cfg = Config.from_yaml(config)
        click.echo(f"Loaded config from: {config}")
    else:
        cfg = Config.from_env()

    # Override with CLI options (CLI takes precedence)
    training_config = cfg.training
    if score_top_percentile is not None:
        training_config.score_top_percentile = score_top_percentile
    if score_name is not None:
        training_config.score_name = score_name
    # Note: max_sessions and sample_size are no longer supported via CLI.
    # They should be specified when generating dataset.json via generate_manifest.py
    if summarization_method is not None:
        training_config.summarization_method = SummarizationMethod(summarization_method.lower())
    if embedding_model is not None:
        training_config.embedding_model = embedding_model
    if umap_n_neighbors is not None:
        training_config.umap_n_neighbors = umap_n_neighbors
    if umap_n_components is not None:
        training_config.umap_n_components = umap_n_components
    if min_cluster_size is not None:
        training_config.min_cluster_size = min_cluster_size
    if min_rubrics is not None:
        training_config.min_rubrics = min_rubrics
    if max_rubrics is not None:
        training_config.max_rubrics = max_rubrics

    # Resolve paths (CLI overrides config)
    dataset_dir = dataset_dir or cfg.dataset_dir
    dataset_manifest = dataset_manifest or cfg.dataset_manifest
    # Default output to ./output directory
    if output is None:
        if cfg.output_path:
            # If config has output_path, check if it's a file or directory
            if cfg.output_path.suffix == ".json":
                output = cfg.output_path.parent / "output"
            else:
                output = cfg.output_path
        else:
            output = Path("output")
    prompts_dir = prompts_dir or cfg.prompts_dir

    # Validate required paths
    if not dataset_dir:
        raise click.UsageError("--dataset-dir is required (or provide via config)")
    if not dataset_manifest:
        raise click.UsageError("--dataset-manifest is required (or provide via config)")

    if training_config.score_top_percentile is not None:
        filter_desc = (
            f"top {training_config.score_top_percentile}% {training_config.score_name} scores"
        )
    else:
        filter_desc = "no score filter (all sessions)"

    click.echo(f"Starting training with filter: {filter_desc}")
    click.echo(f"Dataset directory: {dataset_dir}")
    click.echo(f"Manifest: {dataset_manifest}")
    click.echo(f"Summarization method: {training_config.summarization_method.value}")

    trainer = Trainer(
        dataset_dir=dataset_dir,
        manifest_path=dataset_manifest,
        prompts_dir=prompts_dir,
        config=training_config,
        extraction_llm_config=cfg.extraction_llm,
        summarization_llm_config=cfg.summarization_llm,
        full_config=cfg,
        rate_limiter_config=cfg.rate_limiter,
    )

    try:

        async def run_training():
            nonlocal output
            with get_usage_metadata_callback() as cb:
                rubrics, raw_rubrics_map, _ = await trainer.train()

                if not rubrics.rubrics:
                    click.echo("Warning: No rubrics were generated. Check your dataset.", err=True)
                    sys.exit(1)

                # Ensure output is a directory (default to ./output)
                if output is None:
                    output = Path("output")
                elif output.suffix == ".json":
                    # If user provided a .json file, use its parent directory
                    output = output.parent / "output"

                result_folder = await trainer.save_rubrics(rubrics, output, raw_rubrics_map)
                click.echo(f"Generated {len(rubrics.rubrics)} rubrics")
                click.echo(f"Saved to: {result_folder}")

                # Display rubric summary
                click.echo("\nGenerated Rubrics:")
                for rubric in rubrics.rubrics:
                    click.echo(f"  - {rubric.id}: {rubric.name}")

                # Display token usage
                if cb.usage_metadata:
                    format_token_usage(cb.usage_metadata)

        asyncio.run(run_training())

    except Exception as e:
        click.echo(f"Error during training: {e}", err=True)
        if ctx.obj.get("verbose"):
            import traceback

            traceback.print_exc()
        sys.exit(1)


@main.command()
@click.option(
    "--rubrics",
    "-r",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to rubrics JSON file",
)
@click.option(
    "--session",
    "-s",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to chat session JSONL file",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("result.json"),
    help="Output path for evaluation result (default: result.json)",
)
@click.option(
    "--prompts-dir",
    "-p",
    type=click.Path(exists=True, path_type=Path),
    default=Path("prompts"),
    help="Directory containing prompt templates (default: prompts/)",
)
@click.pass_context
def evaluate(
    ctx: click.Context,
    rubrics: Path,
    session: Path,
    output: Path,
    prompts_dir: Path,
) -> None:
    """Evaluate a chat session against learned rubrics."""
    click.echo(f"Loading rubrics from: {rubrics}")
    click.echo(f"Evaluating session: {session}")

    try:
        # Load rubrics
        rubric_list = RubricList.from_json(rubrics)
        click.echo(f"Loaded {len(rubric_list.rubrics)} rubrics")

        # Load session
        chat_session = ChatSession.from_jsonl(session)
        click.echo(f"Loaded session {chat_session.session_id} ({chat_session.turn_count} turns)")

        # Create evaluator and run
        cfg = Config.from_env()
        evaluator = Evaluator(
            prompts_dir=prompts_dir,
            rate_limiter_config=cfg.rate_limiter,
        )

        with get_usage_metadata_callback() as cb:
            result = asyncio.run(
                evaluator.evaluate(
                    rubrics=rubric_list.rubrics,
                    session=chat_session,
                    rubrics_version=rubric_list.version,
                )
            )

            # Save result
            evaluator.save_result(result, output)
            click.echo(f"Saved result to: {output}")

            # Display summary
            click.echo(f"\nEvaluation Summary:")
            click.echo(f"  Session ID: {result.session_id}")
            click.echo(
                f"  Total Score: {result.summary.total_score}/{result.summary.max_score} "
                f"({result.summary.percentage}%)"
            )
            click.echo(f"\nPer-Rubric Scores:")
            for score in result.rubric_scores:
                click.echo(f"  - {score.rubric_name}: {score.score}/{score.max_score}")

            # Display token usage
            if cb.usage_metadata:
                format_token_usage(cb.usage_metadata)

    except Exception as e:
        click.echo(f"Error during evaluation: {e}", err=True)
        if ctx.obj.get("verbose"):
            import traceback

            traceback.print_exc()
        sys.exit(1)


@main.command("evaluate-batch")
@click.option(
    "--rubrics",
    "-r",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to rubrics JSON file",
)
@click.option(
    "--sessions-dir",
    "-s",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing session JSONL files",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("results"),
    help="Output directory for evaluation results (default: results/)",
)
@click.option(
    "--prompts-dir",
    "-p",
    type=click.Path(exists=True, path_type=Path),
    default=Path("prompts"),
    help="Directory containing prompt templates (default: prompts/)",
)
@click.option(
    "--pattern",
    type=str,
    default="*.jsonl",
    help="Glob pattern for session files (default: *.jsonl)",
)
@click.pass_context
def evaluate_batch(
    ctx: click.Context,
    rubrics: Path,
    sessions_dir: Path,
    output_dir: Path,
    prompts_dir: Path,
    pattern: str,
) -> None:
    """Evaluate multiple chat sessions in batch."""
    click.echo(f"Loading rubrics from: {rubrics}")
    click.echo(f"Sessions directory: {sessions_dir}")

    try:
        # Load rubrics
        rubric_list = RubricList.from_json(rubrics)
        click.echo(f"Loaded {len(rubric_list.rubrics)} rubrics")

        # Find session files
        session_files = list(sessions_dir.glob(pattern))
        if not session_files:
            click.echo(f"No session files found matching pattern: {pattern}", err=True)
            sys.exit(1)

        click.echo(f"Found {len(session_files)} session files")

        # Load sessions
        sessions = []
        for sf in session_files:
            try:
                session = ChatSession.from_jsonl(sf)
                sessions.append(session)
            except Exception as e:
                click.echo(f"Warning: Failed to load {sf}: {e}", err=True)

        click.echo(f"Successfully loaded {len(sessions)} sessions")

        # Create evaluator and run
        cfg = Config.from_env()
        evaluator = Evaluator(
            prompts_dir=prompts_dir,
            rate_limiter_config=cfg.rate_limiter,
        )

        with get_usage_metadata_callback() as cb:
            results = asyncio.run(
                evaluator.evaluate_batch(
                    rubrics=rubric_list.rubrics,
                    sessions=sessions,
                    rubrics_version=rubric_list.version,
                )
            )

            # Save results
            rubric_names = {r.id: r.name for r in rubric_list.rubrics}
            summary_path = evaluator.save_batch_results(results, output_dir, rubric_names)

            click.echo(f"\nBatch evaluation complete!")
            click.echo(f"Results saved to: {output_dir}")
            click.echo(f"Summary: {summary_path}")

            # Display summary statistics
            from .models.evaluation import BatchEvaluationSummary

            summary = BatchEvaluationSummary.from_results(results, rubric_names)
            click.echo(f"\nBatch Summary:")
            click.echo(f"  Sessions evaluated: {summary.total_sessions}")
            click.echo(f"  Average score: {summary.average_score}")
            click.echo(f"  Median score: {summary.median_score}")

            # Display token usage
            if cb.usage_metadata:
                format_token_usage(cb.usage_metadata)

    except Exception as e:
        click.echo(f"Error during batch evaluation: {e}", err=True)
        if ctx.obj.get("verbose"):
            import traceback

            traceback.print_exc()
        sys.exit(1)


@main.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to YAML config file (overrides other options)",
)
@click.option(
    "--dataset-dir",
    "-d",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Directory containing JSONL session files",
)
@click.option(
    "--dataset-manifest",
    "-m",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to dataset manifest JSON file",
)
@click.option(
    "--rubrics",
    "-r",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to rubrics JSON file",
)
@click.option(
    "--score-name",
    "-n",
    type=str,
    default=None,
    help="Name of the score to use for comparison (default: 'default')",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output path for validation report (default: validation_report.json)",
)
@click.option(
    "--prompts-dir",
    "-p",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Directory containing prompt templates (default: prompts/)",
)
@click.pass_context
def validate(
    ctx: click.Context,
    config: Path | None,
    dataset_dir: Path | None,
    dataset_manifest: Path | None,
    rubrics: Path | None,
    score_name: str | None,
    output: Path | None,
    prompts_dir: Path | None,
) -> None:
    """Validate evaluation accuracy by comparing predicted vs real scores.

    This command evaluates each session in the dataset using the provided rubrics,
    then compares the predicted scores against the actual scores from the manifest.
    It outputs a validation report with metrics like MAE, RMSE, and correlation.
    """
    # Load config from YAML if provided
    if config:
        cfg = Config.from_yaml(config)
        click.echo(f"Loaded config from: {config}")
    else:
        cfg = Config.from_env()

    # Override with CLI options (CLI takes precedence)
    score_name = score_name or cfg.training.score_name
    dataset_dir = dataset_dir or cfg.dataset_dir
    dataset_manifest = dataset_manifest or cfg.dataset_manifest
    rubrics = rubrics or cfg.rubrics_path
    output = output or cfg.output_path or Path("validation_report.json")
    prompts_dir = prompts_dir or cfg.prompts_dir

    # Validate required paths
    if not dataset_dir:
        raise click.UsageError("--dataset-dir is required (or provide via config)")
    if not dataset_manifest:
        raise click.UsageError("--dataset-manifest is required (or provide via config)")
    if not rubrics:
        raise click.UsageError("--rubrics is required (or provide via config)")

    click.echo(f"Starting validation with score type: '{score_name}'")
    click.echo(f"Dataset directory: {dataset_dir}")
    click.echo(f"Manifest: {dataset_manifest}")
    click.echo(f"Rubrics: {rubrics}")

    validator = Validator(
        dataset_dir=dataset_dir,
        manifest_path=dataset_manifest,
        rubrics_path=rubrics,
        prompts_dir=prompts_dir,
        score_name=score_name,
        llm_config=cfg.evaluation_llm,
        rate_limiter_config=cfg.rate_limiter,
    )

    try:
        # Load rubrics to show count
        rubric_list = RubricList.from_json(rubrics)
        click.echo(f"Loaded {len(rubric_list.rubrics)} rubrics")

        # Run validation
        with get_usage_metadata_callback() as cb:
            report = asyncio.run(validator.validate())

            if report.total_sessions == 0:
                click.echo(
                    f"Warning: No sessions found with score '{score_name}'. "
                    "Check your dataset manifest.",
                    err=True,
                )
                sys.exit(1)

            # Save report
            validator.save_report(report, output)
            click.echo(f"Saved validation report to: {output}")

            # Display summary
            click.echo(f"\nValidation Summary:")
            click.echo(f"  Sessions validated: {report.total_sessions}")
            click.echo(f"  Score type: {report.score_name}")

            click.echo(f"\nMetrics:")
            click.echo(f"  Mean Absolute Error (MAE): {report.metrics.mean_absolute_error}")
            click.echo(
                f"  Root Mean Squared Error (RMSE): {report.metrics.root_mean_squared_error}"
            )
            click.echo(f"  Mean Error (bias): {report.metrics.mean_error}")
            if report.metrics.correlation is not None:
                click.echo(f"  Correlation: {report.metrics.correlation}")
            if report.metrics.r_squared is not None:
                click.echo(f"  R-squared: {report.metrics.r_squared}")

            click.echo(f"\nError Range:")
            click.echo(f"  Min error: {report.metrics.min_error}")
            click.echo(f"  Max error: {report.metrics.max_error}")

            # Show per-session summary if verbose or few sessions
            if ctx.obj.get("verbose") or report.total_sessions <= 10:
                click.echo(f"\nPer-Session Results:")
                for result in report.session_results:
                    click.echo(
                        f"  - {result.file}: predicted={result.predicted_score}, "
                        f"real={result.real_score}, error={result.error:+.4f}"
                    )

            # Display token usage
            if cb.usage_metadata:
                format_token_usage(cb.usage_metadata)

    except Exception as e:
        click.echo(f"Error during validation: {e}", err=True)
        if ctx.obj.get("verbose"):
            import traceback

            traceback.print_exc()
        sys.exit(1)


@main.command()
@click.argument(
    "raw_rubrics",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to YAML config file (optional, uses defaults if not provided)",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for rubrics.json (defaults to same directory as raw-rubrics.json)",
)
@click.option(
    "--method",
    type=click.Choice(["llm", "cluster", "both"], case_sensitive=False),
    default="both",
    help="Summarization method: 'llm' (LLM only), 'cluster' (clustering only), or 'both' (default)",
)
@click.option(
    "--min-rubrics",
    type=int,
    default=None,
    help="Minimum number of rubrics in final list (overrides config)",
)
@click.option(
    "--max-rubrics",
    type=int,
    default=None,
    help="Maximum number of rubrics in final list (overrides config)",
)
@click.option(
    "--prompts-dir",
    "-p",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Directory containing prompt templates (defaults to prompts/)",
)
@click.option(
    "--embedding-model",
    type=str,
    default=None,
    help="Embedding model for clustering (overrides config)",
)
@click.option(
    "--umap-n-neighbors",
    type=int,
    default=None,
    help="UMAP n_neighbors parameter for clustering (overrides config)",
)
@click.option(
    "--umap-n-components",
    type=int,
    default=None,
    help="UMAP n_components parameter for clustering (overrides config)",
)
@click.option(
    "--min-cluster-size",
    type=int,
    default=None,
    help="HDBSCAN min_cluster_size parameter for clustering (overrides config)",
)
@click.pass_context
def summarize(
    ctx: click.Context,
    raw_rubrics: Path,
    config: Path | None,
    output_dir: Path | None,
    method: str,
    min_rubrics: int | None,
    max_rubrics: int | None,
    prompts_dir: Path | None,
    embedding_model: str | None,
    umap_n_neighbors: int | None,
    umap_n_components: int | None,
    min_cluster_size: int | None,
) -> None:
    """Run LLM and/or clustering summarizers on raw rubrics.

    Takes a raw-rubrics.json file (typically from training output) and runs
    one or both summarization methods on it. Useful for comparing different
    summarization approaches without re-running rubric extraction.

    Outputs:
    - rubrics_llm.json (if method is 'llm' or 'both')
    - rubrics_cluster.json (if method is 'cluster' or 'both')
    """
    # Load config
    if config:
        cfg = Config.from_yaml(config)
        click.echo(f"Loaded config from: {config}")
    else:
        cfg = Config.from_env()

    # Determine output directory
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = raw_rubrics.parent

    # Determine prompts directory
    if prompts_dir:
        prompts_dir = prompts_dir
    else:
        prompts_dir = cfg.prompts_dir

    # Override config values from command line
    min_rubrics_val = min_rubrics if min_rubrics is not None else cfg.training.min_rubrics
    max_rubrics_val = max_rubrics if max_rubrics is not None else cfg.training.max_rubrics
    embedding_model_val = embedding_model if embedding_model is not None else cfg.training.embedding_model
    umap_n_neighbors_val = umap_n_neighbors if umap_n_neighbors is not None else cfg.training.umap_n_neighbors
    umap_n_components_val = umap_n_components if umap_n_components is not None else cfg.training.umap_n_components
    min_cluster_size_val = min_cluster_size if min_cluster_size is not None else cfg.training.min_cluster_size

    click.echo(f"Summarization settings: min_rubrics={min_rubrics_val}, max_rubrics={max_rubrics_val}")
    click.echo(f"Method: {method}")
    if method.lower() in ["cluster", "both"]:
        click.echo(f"Clustering settings: embedding_model={embedding_model_val}, "
                   f"umap_n_neighbors={umap_n_neighbors_val}, "
                   f"umap_n_components={umap_n_components_val}, "
                   f"min_cluster_size={min_cluster_size_val}")

    # Load raw rubrics
    def load_raw_rubrics(raw_rubrics_path: Path) -> list[list[Rubric]]:
        """Load raw rubrics from JSON file and convert to list of Rubric lists."""
        with open(raw_rubrics_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        rubric_lists = []
        for session_id, rubric_dicts in raw_data.items():
            rubrics = []
            for rubric_dict in rubric_dicts:
                try:
                    rubric = Rubric(**rubric_dict)
                    rubrics.append(rubric)
                except Exception as e:
                    click.echo(f"Warning: Failed to parse rubric from session {session_id}: {e}", err=True)
                    continue
            if rubrics:
                rubric_lists.append(rubrics)

        total_rubrics = sum(len(r) for r in rubric_lists)
        click.echo(f"Loaded {len(rubric_lists)} sessions with {total_rubrics} total rubrics from {raw_rubrics_path}")
        return rubric_lists

    try:
        rubric_lists = load_raw_rubrics(raw_rubrics)

        if not rubric_lists:
            click.echo("Error: No rubrics found in raw-rubrics.json file", err=True)
            sys.exit(1)

        async def run_summarization():
            llm_rubrics, llm_notes = [], ""
            cluster_rubrics, cluster_notes = [], ""

            # Initialize and run summarizers based on method
            if method.lower() in ["llm", "both"]:
                click.echo("Initializing LLM summarizer...")
                llm_client = GeminiClient(
                    cfg.summarization_llm,
                    rate_limiter_config=cfg.rate_limiter,
                )
                llm_summarizer = RubricSummarizer(
                    llm_client=llm_client,
                    prompt_template_path=Path("rubric_summarizer.txt"),
                    prompts_dir=prompts_dir,
                    min_rubrics=min_rubrics_val,
                    max_rubrics=max_rubrics_val,
                )

            if method.lower() in ["cluster", "both"]:
                click.echo("Initializing clustering summarizer...")
                cluster_summarizer = SemanticClusteringSummarizer(
                    embedding_model=embedding_model_val,
                    umap_n_neighbors=umap_n_neighbors_val,
                    umap_n_components=umap_n_components_val,
                    min_cluster_size=min_cluster_size_val,
                    min_rubrics=min_rubrics_val,
                    max_rubrics=max_rubrics_val,
                )

            # Run summarizations
            if method.lower() == "both":
                click.echo("Starting summarization (both LLM and Clustering)...")
                import asyncio as aio
                llm_task = aio.create_task(llm_summarizer.summarize(rubric_lists))
                cluster_task = aio.create_task(cluster_summarizer.summarize(rubric_lists))
                (llm_rubrics, llm_notes), (cluster_rubrics, cluster_notes) = await aio.gather(
                    llm_task, cluster_task
                )
            elif method.lower() == "llm":
                click.echo("Starting LLM summarization...")
                llm_rubrics, llm_notes = await llm_summarizer.summarize(rubric_lists)
            elif method.lower() == "cluster":
                click.echo("Starting clustering summarization...")
                cluster_rubrics, cluster_notes = await cluster_summarizer.summarize(rubric_lists)

            # Save results
            if method.lower() in ["llm", "both"] and llm_rubrics:
                click.echo(f"LLM summarization complete. Generated {len(llm_rubrics)} final rubrics")
                llm_rubric_list = RubricList(
                    version="1.0",
                    created_at=datetime.utcnow(),
                    training_config=None,
                    rubrics=llm_rubrics,
                    consolidation_notes=llm_notes,
                )
                llm_output_path = output_dir / "rubrics_llm.json"
                llm_rubric_list.to_json(llm_output_path)
                click.echo(f"Saved LLM rubrics to {llm_output_path}")

            if method.lower() in ["cluster", "both"] and cluster_rubrics:
                click.echo(f"Clustering summarization complete. Generated {len(cluster_rubrics)} final rubrics")
                cluster_rubric_list = RubricList(
                    version="1.0",
                    created_at=datetime.utcnow(),
                    training_config=None,
                    rubrics=cluster_rubrics,
                    consolidation_notes=cluster_notes,
                )
                cluster_output_path = output_dir / "rubrics_cluster.json"
                cluster_rubric_list.to_json(cluster_output_path)
                click.echo(f"Saved clustering rubrics to {cluster_output_path}")

                # Generate visualization if clustering data is available
                if (
                    cluster_summarizer.last_embeddings is not None
                    and cluster_summarizer.last_clusters is not None
                    and cluster_summarizer.last_all_rubrics is not None
                ):
                    try:
                        save_clustering_visualization(
                            output_dir,
                            cluster_summarizer.last_embeddings,
                            cluster_summarizer.last_clusters,
                            cluster_summarizer.last_all_rubrics,
                            cluster_summarizer.last_cluster_info,
                            umap_embeddings=cluster_summarizer.last_umap_embeddings,
                        )
                        click.echo(f"Saved clustering visualization to {output_dir}/clustering_visualization.png")
                    except Exception as e:
                        click.echo(f"Warning: Failed to generate clustering visualization: {e}", err=True)

        asyncio.run(run_summarization())
        click.echo(f"Done! Summarization complete using method: {method}")

    except Exception as e:
        click.echo(f"Error during summarization: {e}", err=True)
        if ctx.obj.get("verbose"):
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
