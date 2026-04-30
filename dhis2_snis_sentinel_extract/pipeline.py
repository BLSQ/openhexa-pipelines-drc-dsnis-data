import json
from pathlib import Path

import polars as pl
from d2d_development.extract import DHIS2Extractor
from openhexa.sdk import current_run, parameter, pipeline, workspace
from openhexa.toolbox.dhis2 import DHIS2
from openhexa.toolbox.dhis2.dataframe import get_datasets
from utils import (
    add_files_to_dataset,
    connect_to_dhis2,
    get_extract_periods,
    load_configuration,
    resolve_dates_and_validate,
)


@pipeline("dhis2-snis-sentinel-extract")
@parameter(
    code="start_date",
    name="Start date (format: YYYYMM)",
    default=None,
    type=str,
    required=False,
    help=(
        "Start date for data extraction in YYYYMM format. "
        "If not set, it will default to current date minus NUMBER_MONTHS_WINDOW."
    ),
)
@parameter(
    code="end_date",
    name="End date (format: YYYYMM)",
    default=None,
    type=str,
    required=False,
    help=("End date for data extraction in YYYYMM format. If not set, it will default to current date minus 1."),
)
@parameter(
    code="run_extract_data",
    name="Extract data",
    type=bool,
    default=True,
    help="Extract data elements from source DHIS2.",
)
@parameter(
    code="add_to_dataset",
    name="Add extracted data to dataset.",
    type=bool,
    default=True,
    help="Add extracts created in this run to the  dataset.",
)
def dhis2_snis_sentinel_extract(start_date: str, end_date: str, run_extract_data: bool, add_to_dataset: bool):
    """Main function for the pipeline."""
    current_run.log_info("Starting DHIS2 SNIS Sentinel Extract pipeline execution.")
    pipeline_path = Path(workspace.files_path) / "pipelines" / "dhis2_snis_sentinel_extract"

    try:
        updates_collector = {}

        # Extracts only at DHIS2 dataset level.
        extract_data(
            pipeline_path=pipeline_path,
            start_date=start_date,
            end_date=end_date,
            run_task=run_extract_data,
            updates_collector=updates_collector,
        )

        update_snis_dataset_with_extracts(
            pipeline_path=pipeline_path,
            updates_collector=updates_collector,
            dataset_id="snis-data-elements-extracts",
            run_task=add_to_dataset,
        )
        current_run.log_info("Pipeline execution completed successfully.")

    except Exception as e:
        current_run.log_error(f"An error occurred: {e}")
        raise


def extract_data(
    pipeline_path: str,
    start_date: str,
    end_date: str,
    run_task: bool,
    updates_collector: dict,
) -> None:
    """Data extraction task."""
    if not run_task:
        return

    current_run.log_info("Retrieving DHIS2 analytics data")

    config = load_configuration(pipeline_path / "config" / "extract_config.json")
    dhis2_snis_client = connect_to_dhis2(connection_str=config["SETTINGS"]["DHIS2_CONNECTION"])

    # get dates and validate
    start, end = resolve_dates_and_validate(start_date, end_date, config)
    extract_periods = get_extract_periods(start, end)

    current_run.log_info(f"Download MODE: {config['SETTINGS']['MODE']} from: {start} to {end}")

    # limits
    dhis2_snis_client.analytics.MAX_DX = 100
    dhis2_snis_client.analytics.MAX_ORG_UNITS = 100
    dhis2_snis_client.data_value_sets.MAX_DATA_ELEMENTS = 100
    dhis2_snis_client.data_value_sets.MAX_ORG_UNITS = 100

    handle_data_element_extracts(
        pipeline_path=pipeline_path,
        dhis2_client=dhis2_snis_client,
        config=config,
        extract_periods=extract_periods,
        updates_collector=updates_collector,
    )

    current_run.log_info("Extracts finished.")


def handle_data_element_extracts(
    pipeline_path: Path,
    dhis2_client: DHIS2,
    config: dict,
    extract_periods: list,
    updates_collector: dict[Path],
) -> None:
    """Handles data elements extracts based on the configuration."""
    data_element_extracts = config.get("DATA_ELEMENTS", {}).get("EXTRACTS", [])
    if not data_element_extracts:
        current_run.log_info("No data element to extract.")
        return

    current_run.log_info("Starting data element extracts.")

    source_datasets = get_datasets(dhis2_client)
    dhis2_extractor = DHIS2Extractor(
        dhis2_client=dhis2_client, download_mode=config.get("SETTINGS", {}).get("MODE", "DOWNLOAD_REPLACE")
    )
    # loop over the available extract configurations
    for idx, extract in enumerate(data_element_extracts):
        extract_id = extract.get("EXTRACT_UID")
        dataset_id = extract.get("DATASET_UID", None)
        data_element_uids = extract.get("UIDS", [])

        if not extract_id:
            current_run.log_warning(f"No 'EXTRACT_UID' defined at position: {idx}. This is required, extract skipped.")
            continue

        if not dataset_id:
            current_run.log_warning(f"No 'DATASET_UID' defined for extract: {extract_id}, extract skipped.")
            continue

        if not data_element_uids:
            current_run.log_warning(f"No data elements defined for extract: {extract_id}, extract skipped.")
            continue

        source_dataset = source_datasets.filter(pl.col("id").is_in([dataset_id]))
        org_units = list(source_dataset["organisation_units"][0])
        current_run.log_info(
            f"Starting data elements extract ID: '{extract_id}' ({idx + 1}) "
            f"with {len(data_element_uids)} data elements across {len(org_units)} org units from dataset "
            f"'{source_dataset['name'][0]}' ({dataset_id})."
        )

        for period in extract_periods:
            try:
                extract_path = dhis2_extractor.data_elements.download_period(
                    data_elements=data_element_uids,
                    org_units=org_units,
                    period=period,
                    output_dir=pipeline_path / "data" / "extracts" / "data_elements" / f"extract_{extract_id}",
                    filename=f"data_{extract_id}_{period}.parquet",
                )
                if extract_path:
                    updates_collector.setdefault(extract_id, []).append(extract_path)

            except Exception:
                current_run.log_error(
                    f"Extract {extract_id} download failed for period {period}, skipping to next extract."
                )
                break  # skip to next extract

        current_run.log_info(f"Extract {extract_id} finished.")


def update_snis_dataset_with_extracts(
    pipeline_path: Path, updates_collector: dict[Path], dataset_id: str, run_task: bool
) -> None:
    """Updates the SNIS dataset with the new extracts.

    This function takes the paths of the new extracts from the updates collector and updates the OH dataset.
    NOTE: Additionally includes a json file to link the extract files with their extract_id required for integration.
    """
    if not run_task:
        return

    new_extracts = [item for values in updates_collector.values() for item in values]

    if not new_extracts:
        current_run.log_info("No new extracts to update in the dataset.")
        return

    mapping_file_path = pipeline_path / "data" / "updates_collector.json"
    save_updates_collector_json(
        updates_collector=updates_collector, output_path=pipeline_path / "data" / "updates_collector.json"
    )

    try:
        add_files_to_dataset(
            dataset_id=dataset_id,
            file_paths=new_extracts + [mapping_file_path],
            ds_version_prefix="SNIS_SENTINEL_extract",
        )
    except Exception as e:
        raise Exception(f"Error while updating SNIS dataset: {e}") from e


def save_updates_collector_json(updates_collector: dict, output_path: Path) -> None:
    """Save updates_collector as a JSON file, raising an error if it fails."""
    try:
        serializable = {k: [str(p.name) for p in v] for k, v in updates_collector.items()}
        with Path.open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
    except Exception as e:
        raise RuntimeError(f"Failed to save updates_collector to {output_path}: {e}") from e


if __name__ == "__main__":
    dhis2_snis_sentinel_extract()
