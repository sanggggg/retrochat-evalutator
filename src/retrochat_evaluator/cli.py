"""CLI entry points for Retrochat Evaluator."""

import asyncio
import logging
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

from .config import Config, TrainingConfig, EvaluationConfig, LLMConfig
from .models.rubric import RubricList
from .models.chat_session import ChatSession
from .training.trainer import Trainer
from .evaluation.evaluator import Evaluator


def setup_logging(verbose: bool) -> None:
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


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
    "--dataset-dir",
    "-d",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing JSONL session files",
)
@click.option(
    "--dataset-manifest",
    "-m",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to dataset manifest JSON file",
)
@click.option(
    "--score-threshold",
    "-t",
    type=float,
    default=4.0,
    help="Minimum score threshold for sessions (default: 4.0)",
)
@click.option(
    "--score-name",
    "-n",
    type=str,
    default="default",
    help="Name of the score to use for filtering (default: 'default')",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("rubrics.json"),
    help="Output path for generated rubrics (default: rubrics.json)",
)
@click.option(
    "--prompts-dir",
    "-p",
    type=click.Path(exists=True, path_type=Path),
    default=Path("prompts"),
    help="Directory containing prompt templates (default: prompts/)",
)
@click.option(
    "--max-sessions",
    type=int,
    default=None,
    help="Maximum number of sessions to use for training",
)
@click.pass_context
def train(
    ctx: click.Context,
    dataset_dir: Path,
    dataset_manifest: Path,
    score_threshold: float,
    score_name: str,
    output: Path,
    prompts_dir: Path,
    max_sessions: int | None,
) -> None:
    """Train evaluation rubrics from high-quality chat sessions."""
    click.echo(f"Starting training with {score_name} score threshold >= {score_threshold}")
    click.echo(f"Dataset directory: {dataset_dir}")
    click.echo(f"Manifest: {dataset_manifest}")

    training_config = TrainingConfig(
        score_threshold=score_threshold,
        score_name=score_name,
        max_sessions=max_sessions,
    )

    trainer = Trainer(
        dataset_dir=dataset_dir,
        manifest_path=dataset_manifest,
        prompts_dir=prompts_dir,
        config=training_config,
    )

    try:
        rubrics = asyncio.run(trainer.train())

        if not rubrics.rubrics:
            click.echo("Warning: No rubrics were generated. Check your dataset.", err=True)
            sys.exit(1)

        trainer.save_rubrics(rubrics, output)
        click.echo(f"Generated {len(rubrics.rubrics)} rubrics")
        click.echo(f"Saved to: {output}")

        # Display rubric summary
        click.echo("\nGenerated Rubrics:")
        for rubric in rubrics.rubrics:
            click.echo(f"  - {rubric.id}: {rubric.name}")

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
        evaluator = Evaluator(prompts_dir=prompts_dir)
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
        evaluator = Evaluator(prompts_dir=prompts_dir)
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

    except Exception as e:
        click.echo(f"Error during batch evaluation: {e}", err=True)
        if ctx.obj.get("verbose"):
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
