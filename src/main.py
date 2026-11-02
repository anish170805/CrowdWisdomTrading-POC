"""Single application entry point for the pipeline.

Run ``python -m src.main --help`` or the installed ``crowdxbot`` command.
This module deliberately contains no pipeline business logic; later modules
add commands that consume the same validated settings object.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Annotated, cast

import typer
from rich.console import Console
from rich.table import Table

from src import __version__
from src.config import ConfigurationError, Settings, get_settings
from src.pipeline import run_pipeline
from src.ingestion.apify_x import (
    ApifyIngestionError,
    CollectionResult,
    PreparedCollection,
    collect_x_posts,
    inspect_actor,
    prepare_x_collection,
)
from src.ingestion.historical_tweets import (
    HistoricalImportError,
    ImportResult,
)
from src.ingestion.historical_tweets import (
    import_historical_tweets as run_historical_import,
)
from src.logging_config import configure_logging, get_logger, new_run_id, run_context

app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    help="X sentiment to asset price prediction pipeline.",
)
console = Console()
err_console = Console(stderr=True)


def _print_configuration_error(error: ConfigurationError) -> None:
    err_console.print("[bold red]Configuration error[/bold red]")
    for issue in error.issues:
        err_console.print(f"  [red]-[/red] {issue}")
    err_console.print(
        "\nNext step: copy [bold].env.example[/bold] to [bold].env[/bold], "
        "fill in the required values, and run [bold]check-config[/bold] again."
    )


def _load_settings() -> Settings:
    try:
        return get_settings()
    except ConfigurationError as error:
        _print_configuration_error(error)
        raise typer.Exit(code=1) from None


def _render_summary(summary: dict[str, object], show_paths: bool) -> None:
    table = Table(title="Configuration OK", title_justify="left", show_header=False, box=None)
    table.add_column("Setting", style="bold cyan", no_wrap=True)
    table.add_column("Value", overflow="fold")

    credentials = cast(dict[str, str], summary["credentials"])
    credential_text = " | ".join(f"{name}: {status}" for name, status in credentials.items())

    table.add_row("Asset", f"{summary['asset_symbol']} ({summary['asset_type']})")
    table.add_row("Timezone", str(summary["timezone"]))
    table.add_row(
        "Date window",
        f"{summary['date_start']} to {summary['date_end']} ({summary['date_window_days']} days)",
    )
    table.add_row("Price provider", str(summary["data_provider"]))
    table.add_row("Apify actor", str(summary["apify_actor_id"]))
    table.add_row("Credentials", credential_text)
    table.add_row("Database", str(summary["database_url"]))
    horizons = summary["model_horizons"]
    horizon_text = (
        ",".join(str(item) for item in horizons) if isinstance(horizons, list) else str(horizons)
    )
    table.add_row(
        "Model",
        f"{summary['model_type']}, horizons {horizon_text}, seed {summary['model_random_seed']}",
    )
    walk_forward = cast(dict[str, int], summary["walk_forward"])
    table.add_row(
        "Walk-forward",
        f"initial {walk_forward['initial_train_days']}d, "
        f"test {walk_forward['test_days']}d, gap {walk_forward['gap_days']}d",
    )
    table.add_row(
        "Logging",
        f"{summary['logging']['level']} ({summary['logging']['format']})",  # type: ignore[index]
    )

    if show_paths:
        paths = cast(dict[str, str], summary["paths"])
        for label, path in paths.items():
            table.add_row(f"Path: {label}", str(path))

    console.print(table)
    console.print("[green]Configuration is valid.[/green]")


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        is_eager=True,
        help="Show the application version and exit.",
    ),
) -> None:
    if version:
        typer.echo(f"crowdxbot {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


@app.command("check-config")
def check_config(
    show_paths: bool = typer.Option(
        True,
        "--paths/--no-paths",
        help="Show resolved output paths in the summary.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit the redacted summary as JSON.",
    ),
) -> None:
    """Validate configuration and create the configured output directories."""
    settings = _load_settings()

    try:
        settings.ensure_directories()
    except ConfigurationError as error:
        _print_configuration_error(error)
        raise typer.Exit(code=1) from None

    configure_logging(settings)

    with run_context(new_run_id("config")) as run_id:
        logger = get_logger("src.main")
        logger.info(
            "Configuration validated",
            extra={"event": "config.validated", "asset_symbol": settings.asset_symbol},
        )

        summary = settings.safe_summary()
        if not show_paths:
            summary.pop("paths", None)

        if json_output:
            typer.echo(json.dumps(summary, indent=2, sort_keys=True))
        else:
            _render_summary(summary, show_paths)
            console.print(f"Run ID: [dim]{run_id}[/dim]")


def _parse_column_overrides(values: list[str] | None) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in values or []:
        if "=" not in item:
            raise typer.BadParameter(
                f"invalid --column value {item!r}; use canonical=source-column"
            )
        field_name, column = item.split("=", 1)
        field_name = field_name.strip()
        column = column.strip()
        if not field_name or not column:
            raise typer.BadParameter(
                f"invalid --column value {item!r}; use canonical=source-column"
            )
        overrides[field_name] = column
    return overrides


def _parse_iso_date(value: str | None, option: str) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        raise typer.BadParameter(f"{option} must be an ISO date (YYYY-MM-DD)") from None


def _render_import_summary(result: ImportResult) -> None:
    summary = result.summary
    table = Table(
        title="Historical tweet import",
        title_justify="left",
        show_header=False,
        box=None,
    )
    table.add_column("Item", style="bold cyan", no_wrap=True)
    table.add_column("Value", overflow="fold")

    table.add_row("Batch", summary.import_batch_id)
    table.add_row("Source", f"{summary.source_filename} ({summary.source_format})")
    table.add_row(
        "Rows",
        f"{summary.total_rows} total, {summary.valid_rows} valid, "
        f"{summary.quarantined_rows} quarantined, {summary.blank_rows_skipped} blank skipped",
    )
    date_range = summary.date_range_utc
    table.add_row(
        "Date range (UTC)",
        f"{date_range.get('start') or '-'} to {date_range.get('end') or '-'}",
    )
    window = summary.configured_window
    if window is None:
        window_text = "not configured"
    else:
        window_text = f"{window['start']} to {window['end']} ({window['timezone']})"
    table.add_row("Configured window", window_text)
    mapped = ", ".join(
        f"{field}={column}"
        for field, column in summary.column_mapping.items()
        if column is not None
    )
    table.add_row("Column mapping", mapped or "none")
    table.add_row("Raw records", summary.raw_records_path)
    table.add_row("Canonical records", summary.canonical_records_path)
    table.add_row("Quarantine", summary.quarantine_path)
    table.add_row("Manifest", summary.manifest_path)
    if summary.source_copy_path:
        table.add_row("Source copy", summary.source_copy_path)
    console.print(table)
    console.print(
        f"[green]Imported {summary.valid_rows} record(s); "
        f"{summary.quarantined_rows} quarantined.[/green]"
    )

    if summary.parse_errors:
        console.print("[yellow]Quarantined rows (first 5):[/yellow]")
        for line in summary.parse_errors[:5]:
            console.print(f"  [yellow]-[/yellow] {line}")
        if summary.parse_errors_truncated:
            console.print("  [yellow]... see the manifest/quarantine file for the rest[/yellow]")

    if summary.warnings:
        console.print("[yellow]Warnings:[/yellow]")
        for line in summary.warnings[:5]:
            console.print(f"  [yellow]-[/yellow] {line}")
        if len(summary.warnings) > 5:
            console.print(f"  [yellow]... {len(summary.warnings) - 5} more[/yellow]")


@app.command("import-historical-tweets")
def import_historical_tweets_command(
    source: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="CSV, TSV, XLSX/XLSM, JSON, or JSON Lines export.",
        ),
    ],
    source_format: Annotated[
        str | None,
        typer.Option(
            "--format",
            "-f",
            help="Override format detection: csv, tsv, xlsx, json, or jsonl.",
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            help="Exact batch directory; defaults to the configured raw-data area.",
        ),
    ] = None,
    columns: Annotated[
        list[str] | None,
        typer.Option(
            "--column",
            "-c",
            help="Map a canonical field to a source column, e.g. --column text=Tweet.",
        ),
    ] = None,
    assume_timezone: Annotated[
        str | None,
        typer.Option(
            "--assume-timezone",
            help="IANA timezone for naive timestamps; defaults to configured TIMEZONE.",
        ),
    ] = None,
    apply_date_window: Annotated[
        bool,
        typer.Option(
            "--apply-date-window",
            help="Quarantine rows outside the configured/overridden date window.",
        ),
    ] = False,
    start_date: Annotated[
        str | None,
        typer.Option(
            "--start-date",
            help="Inclusive ISO start date override (YYYY-MM-DD).",
        ),
    ] = None,
    end_date: Annotated[
        str | None,
        typer.Option(
            "--end-date",
            help="Inclusive ISO end date override (YYYY-MM-DD).",
        ),
    ] = None,
    sheet: Annotated[
        str | None,
        typer.Option(
            "--sheet",
            help="Worksheet name for XLSX input (default: active sheet).",
        ),
    ] = None,
    copy_source: Annotated[
        bool,
        typer.Option(
            "--copy-source/--no-copy-source",
            help="Copy the original export into the batch directory.",
        ),
    ] = True,
    fail_on_quarantine: Annotated[
        bool,
        typer.Option(
            "--fail-on-quarantine",
            help="Exit with status 2 when any row is quarantined.",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Emit the import summary as JSON.",
        ),
    ] = False,
) -> None:
    """Import an explicit historical X export and quarantine malformed rows."""
    settings = _load_settings()

    try:
        settings.ensure_directories()
    except ConfigurationError as error:
        _print_configuration_error(error)
        raise typer.Exit(code=1) from None

    configure_logging(settings)
    overrides = _parse_column_overrides(columns)
    parsed_start = _parse_iso_date(start_date, "--start-date")
    parsed_end = _parse_iso_date(end_date, "--end-date")

    with run_context(new_run_id("historical-import")) as run_id:
        try:
            result = run_historical_import(
                source,
                settings=settings,
                output_dir=output_dir,
                source_format=source_format,
                column_map=overrides,
                assume_timezone=assume_timezone,
                apply_date_window=apply_date_window,
                start_date=parsed_start,
                end_date=parsed_end,
                sheet_name=sheet,
                copy_source=copy_source,
            )
        except HistoricalImportError as error:
            err_console.print("[bold red]Historical tweet import failed[/bold red]")
            err_console.print(str(error))
            raise typer.Exit(code=1) from None

        if json_output:
            typer.echo(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        else:
            _render_import_summary(result)
            console.print(f"Run ID: [dim]{run_id}[/dim]")

    if fail_on_quarantine and result.summary.quarantined_rows:
        err_console.print(f"[red]{result.summary.quarantined_rows} row(s) were quarantined.[/red]")
        raise typer.Exit(code=2)


def _render_prepared_collection(prepared: PreparedCollection) -> None:
    query = prepared.query
    table = Table(
        title="Apify X request (dry run)",
        title_justify="left",
        show_header=False,
        box=None,
    )
    table.add_column("Item", style="bold cyan", no_wrap=True)
    table.add_column("Value", overflow="fold")
    table.add_row("Asset", f"{query.asset_symbol} ({query.asset_type})")
    table.add_row("Query", query.query)
    table.add_row(
        "Window",
        f"{query.start_date.isoformat()} to {query.end_date.isoformat()} ({query.timezone})",
    )
    table.add_row("Max items", str(prepared.max_items))
    table.add_row("Template", prepared.template_source)
    table.add_row("Placeholders", ", ".join(prepared.used_placeholders))
    table.add_row(
        "Local schema",
        prepared.input_schema_path or "not configured",
    )
    console.print(table)
    console.print("[bold]Rendered actor input[/bold]")
    console.print_json(data=prepared.run_input)


def _render_collection_summary(result: CollectionResult) -> None:
    manifest = result.manifest
    table = Table(
        title="Apify X collection",
        title_justify="left",
        show_header=False,
        box=None,
    )
    table.add_column("Item", style="bold cyan", no_wrap=True)
    table.add_column("Value", overflow="fold")

    table.add_row("Collection", manifest.collection_id)
    table.add_row("Actor", manifest.actor_id)
    table.add_row("Run", manifest.actor_run_id or "-")
    table.add_row("Dataset", manifest.dataset_id or "-")
    table.add_row("Status", manifest.run_status)
    table.add_row("Query", manifest.query)
    window = manifest.collection_window
    table.add_row(
        "Window",
        f"{window['start_date']} to {window['end_date']} ({window['timezone']})",
    )
    table.add_row(
        "Items",
        f"{manifest.dataset_item_count} dataset, {manifest.valid_records} valid, "
        f"{manifest.quarantined_records} quarantined",
    )
    table.add_row(
        "Flags",
        f"partial={manifest.partial}, empty={manifest.empty_result}, "
        f"truncated={manifest.truncated}",
    )
    table.add_row(
        "Retries",
        f"{manifest.api_retries} API ({manifest.rate_limit_retries} rate-limit)",
    )
    table.add_row("Records", manifest.records_path)
    table.add_row("Quarantine", manifest.quarantine_path)
    table.add_row("Raw dataset", manifest.dataset_items_path)
    table.add_row("Run metadata", manifest.actor_run_path)
    table.add_row("Manifest", manifest.manifest_path)
    console.print(table)

    if manifest.empty_result:
        console.print(
            "[yellow]The run succeeded but returned an empty dataset. "
            "Check the query and actor input.[/yellow]"
        )
    else:
        console.print(
            f"[green]Collected {manifest.valid_records} record(s); "
            f"{manifest.quarantined_records} quarantined.[/green]"
        )

    if manifest.parse_errors:
        console.print("[yellow]Quarantined items (first 5):[/yellow]")
        for line in manifest.parse_errors[:5]:
            console.print(f"  [yellow]-[/yellow] {line}")
        if manifest.parse_errors_truncated:
            console.print("  [yellow]... see the manifest/quarantine file for the rest[/yellow]")

    if manifest.warnings:
        console.print("[yellow]Warnings:[/yellow]")
        for line in manifest.warnings[:5]:
            console.print(f"  [yellow]-[/yellow] {line}")
        if len(manifest.warnings) > 5:
            console.print(f"  [yellow]... {len(manifest.warnings) - 5} more[/yellow]")


@app.command("ingest-x")
def ingest_x_command(
    start_date: Annotated[
        str | None,
        typer.Option("--start-date", help="Inclusive ISO start date override (YYYY-MM-DD)."),
    ] = None,
    end_date: Annotated[
        str | None,
        typer.Option("--end-date", help="Inclusive ISO end date override (YYYY-MM-DD)."),
    ] = None,
    query_mode: Annotated[
        str | None,
        typer.Option(
            "--query-mode",
            help="cashtag_and_symbol, cashtag_only, or symbol_only.",
        ),
    ] = None,
    extra_terms: Annotated[
        list[str] | None,
        typer.Option(
            "--extra-term",
            "-t",
            help="Extra query alias (repeatable), e.g. --extra-term 'Apple Inc'.",
        ),
    ] = None,
    exclude_terms: Annotated[
        list[str] | None,
        typer.Option(
            "--exclude-term",
            "-x",
            help="Term excluded from the query (repeatable), e.g. --exclude-term giveaway.",
        ),
    ] = None,
    language: Annotated[
        str | None,
        typer.Option("--language", help="Optional X language filter, e.g. en."),
    ] = None,
    max_items: Annotated[
        int | None,
        typer.Option("--max-items", help="Override APIFY_MAX_ITEMS for this run."),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            help="Exact collection directory; defaults to the configured raw-data area.",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Build and validate the query/actor input without starting a run.",
        ),
    ] = False,
    allow_partial: Annotated[
        bool,
        typer.Option(
            "--allow-partial",
            help="Accept records from a failed, aborted, or timed-out run.",
        ),
    ] = False,
    apply_date_window: Annotated[
        bool,
        typer.Option(
            "--apply-date-window",
            help="Quarantine dataset items outside the configured/overridden date window.",
        ),
    ] = False,
    validate_remote: Annotated[
        bool,
        typer.Option(
            "--validate-remote",
            help="Ask Apify to validate the rendered actor input before starting.",
        ),
    ] = False,
    fail_on_quarantine: Annotated[
        bool,
        typer.Option(
            "--fail-on-quarantine",
            help="Exit with status 2 when any item is quarantined.",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit the collection summary as JSON."),
    ] = False,
) -> None:
    """Collect recent X posts through the configured Apify actor."""
    settings = _load_settings()

    try:
        settings.ensure_directories()
    except ConfigurationError as error:
        _print_configuration_error(error)
        raise typer.Exit(code=1) from None

    configure_logging(settings)
    parsed_start = _parse_iso_date(start_date, "--start-date")
    parsed_end = _parse_iso_date(end_date, "--end-date")

    with run_context(new_run_id("apify-x")):
        if dry_run:
            try:
                prepared = prepare_x_collection(
                    settings,
                    start_date=parsed_start,
                    end_date=parsed_end,
                    query_mode=query_mode,
                    extra_terms=extra_terms,
                    exclude_terms=exclude_terms,
                    language=language,
                    max_items=max_items,
                )
            except ApifyIngestionError as error:
                err_console.print("[bold red]Could not prepare the Apify request[/bold red]")
                err_console.print(str(error))
                raise typer.Exit(code=1) from None

            if json_output:
                typer.echo(json.dumps(prepared.to_dict(), indent=2, sort_keys=True))
            else:
                _render_prepared_collection(prepared)
            return

        try:
            result = collect_x_posts(
                settings,
                start_date=parsed_start,
                end_date=parsed_end,
                query_mode=query_mode,
                extra_terms=extra_terms,
                exclude_terms=exclude_terms,
                language=language,
                max_items=max_items,
                output_dir=output_dir,
                validate_remote=validate_remote,
                allow_partial=allow_partial,
                apply_date_window=apply_date_window,
            )
        except ApifyIngestionError as error:
            err_console.print("[bold red]Apify X collection failed[/bold red]")
            err_console.print(str(error))
            raise typer.Exit(code=1) from None

        if json_output:
            typer.echo(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        else:
            _render_collection_summary(result)

    if fail_on_quarantine and result.manifest.quarantined_records:
        err_console.print(
            f"[red]{result.manifest.quarantined_records} item(s) were quarantined.[/red]"
        )
        raise typer.Exit(code=2)


@app.command("inspect-apify-actor")
def inspect_apify_actor_command(
    actor_id: Annotated[
        str | None,
        typer.Option("--actor-id", help="Override APIFY_ACTOR_ID for this inspection."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit the actor metadata as JSON."),
    ] = False,
) -> None:
    """Show Apify actor metadata without starting a run."""
    settings = _load_settings()
    configure_logging(settings)

    try:
        payload = inspect_actor(settings, actor_id=actor_id)
    except ApifyIngestionError as error:
        err_console.print("[bold red]Could not inspect the Apify actor[/bold red]")
        err_console.print(str(error))
        raise typer.Exit(code=1) from None

    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    actor = payload.get("actor")
    actor = actor if isinstance(actor, dict) else {}
    table = Table(title="Apify actor", title_justify="left", show_header=False, box=None)
    table.add_column("Item", style="bold cyan", no_wrap=True)
    table.add_column("Value", overflow="fold")
    table.add_row("Actor ID", str(payload["actor_id"]))
    table.add_row("Name", str(actor.get("name") or "-"))
    table.add_row("Username", str(actor.get("username") or "-"))
    table.add_row("Title", str(actor.get("title") or "-"))
    table.add_row("Modified", str(actor.get("modifiedAt") or "-"))
    console.print(table)
    console.print(
        "Next step: compare the actor's INPUT_SCHEMA.json with "
        "APIFY_INPUT_TEMPLATE, then run [bold]ingest-x --dry-run[/bold]."
    )


@app.command("doctor", hidden=True)
def doctor(
    show_paths: bool = typer.Option(True, "--paths/--no-paths"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Alias for check-config."""
    check_config(show_paths=show_paths, json_output=json_output)


@app.command("version")
def version_command() -> None:
    """Print the application version."""
    typer.echo(f"crowdxbot {__version__}")


@app.command("run-all")
def run_all_command(
    reimport: bool = typer.Option(
        False,
        "--reimport/--no-reimport",
        help="Re-run ingestion stages and rebuild the dataset even if inputs are unchanged.",
    ),
    retrain: bool = typer.Option(
        False,
        "--retrain/--no-retrain",
        help="Retrain the model and re-run walk-forward evaluation even if cached.",
    ),
    force: bool = typer.Option(
        False,
        "--force/--no-force",
        help="Keep going after a failed stage instead of stopping.",
    ),
    historical_file: Path | None = typer.Option(
        None,
        "--historical-file",
        exists=True,
        dir_okay=False,
        help="Historical tweet export to import (skipped if this exact file was already imported).",
    ),
    allow_rule_based_sentiment: bool = typer.Option(
        False,
        "--allow-rule-based-sentiment",
        help="Use keyword scoring if the FinBERT model cannot be loaded (not cached).",
    ),
) -> None:
    """Run the whole pipeline; only stages whose inputs changed are re-run."""
    results = run_pipeline(
        reimport=reimport,
        retrain=retrain,
        force=force,
        historical_file=historical_file,
        allow_rule_based_sentiment=allow_rule_based_sentiment,
    )
    if results.get("error") or results["steps_failed"]:
        raise typer.Exit(code=1)


def main() -> None:
    """Console-script entry point."""
    try:
        app(prog_name="crowdxbot")
    except ConfigurationError as error:
        _print_configuration_error(error)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        err_console.print("\nInterrupted.")
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
