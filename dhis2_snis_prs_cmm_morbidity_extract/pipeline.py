import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import polars as pl
from d2d_development.extract import DHIS2Extractor
from dateutil.relativedelta import relativedelta
from openhexa.sdk import current_run, parameter, pipeline, workspace
from openhexa.toolbox.dhis2 import DHIS2
from openhexa.toolbox.dhis2.dataframe import get_datasets
from utils import (
    add_files_to_dataset,
    connect_to_dhis2,
    get_extract_periods,
    read_json_file,
    resolve_dates_and_validate,
    save_to_parquet,
    select_descendants,
)

# Ticket(s) related to this pipeline:
#   -https://bluesquare.atlassian.net/browse/PATHEOC-412
# github repo:
#   -https://github.com/BLSQ/openhexa-pipelines-drc-dsnis-data


@pipeline("dhis2_snis_prs_cmm_morbidity_extract", timeout=21600)  # 3600 * 6 hours
@parameter(
    code="start_date",
    name="Start date (format: YYYYMM)",
    type=str,
    required=False,
    help=(
        "Start date for data extraction in YYYYMM format. "
        "If not set, it will default to current date minus NUMBER_MONTHS_WINDOW."
    ),
    default=None,
)
@parameter(
    code="end_date",
    name="End date (format: YYYYMM)",
    type=str,
    required=False,
    help=(
        "End date for data extraction in YYYYMM format. "
        "If not set, it will default to current date minus NUMBER_MONTHS_WINDOW."
    ),
    default=None,
)
@parameter(
    code="run_extract_data",
    name="Extract data",
    type=bool,
    default=True,
    help="Extract data elements from source DHIS2.",
)
@parameter(
    code="skip_existing_extracts",
    name="Skip existing extracts",
    type=bool,
    default=False,
    help="Skip downloading extracts that already exist in the repository folder.",
)
@parameter(
    "add_to_dataset",
    name="Add extracts to dataset",
    help="Create a new version with the extracts created in this run.",
    type=bool,
    default=True,
    required=False,
)
def dhis2_snis_prs_cmm_morbidity_extract(
    start_date: str, end_date: str, run_extract_data: bool, skip_existing_extracts: bool, add_to_dataset: bool
):
    """Main pipeline function for DHIS2 dataset synchronization for SNIS -> PRS.

    NOTE: The data is already filtered at this stage to reduce the amount of data to be extracted from SNIS.
    The extracts are created based on the configuration file (extract_config.json) and the pyramid
    selection (sync_config.json).

    Args:
        start_date: Start date for data extraction in YYYYMM format. If not set, it will default to
            current date minus NUMBER_MONTHS_WINDOW (config).
        end_date: End date for data extraction in YYYYMM format. If not set, it will default to
            current date minus 1.
        run_extract_data: If True, runs the pyramid and data extraction tasks.
        skip_existing_extracts: If True, skips downloading extracts (pyramid and data elements)
            that already exist on disk, reusing the existing files instead.
        add_to_dataset: If True, adds the extracted data to the dataset.

    Raises:
        Exception: If an error occurs during the pipeline execution.
    """
    pipeline_path = Path(workspace.files_path) / "pipelines" / "dhis2_snis_prs_cmm_morbidity_extract"
    config = read_json_file(pipeline_path / "configuration" / "extract_config.json")
    source_dhis2 = connect_to_dhis2(connection_str=config["SETTINGS"]["SOURCE_DHIS2_CONNECTION"])
    updates_collector = {}

    try:
        extract_pyramid(
            dhis2_client=source_dhis2,
            sync_config=read_json_file(pipeline_path / "configuration" / "sync_config.json"),
            output_dir=pipeline_path / "data" / "pyramid",
            filename="pyramid_data.parquet",
            updates_collector=updates_collector,
            run_task=run_extract_data,
        )

        extract_prs_data(
            pipeline_path=pipeline_path,
            dhis2_client=source_dhis2,
            config=config,
            start_date=start_date,
            end_date=end_date,
            run_task=run_extract_data,
            updates_collector=updates_collector,
            skip_existing_extracts=skip_existing_extracts,
        )
    except Exception as e:
        current_run.log_error(f"An error occurred data retrieval: {e}")
        raise

    try:
        update_dataset_with_extracts(
            pipeline_path=pipeline_path,
            updates_collector=updates_collector,
            dataset_id="snis-prs-cmm-extract",
            run_task=add_to_dataset,
        )
    except Exception as e:
        current_run.log_error(f"An error occurred while updating the dataset: {e}")
        raise


def extract_pyramid(
    dhis2_client: DHIS2, sync_config: dict, output_dir: Path, filename: str, updates_collector: dict, run_task: bool
) -> None:
    """Extracts the source DHIS2 pyramid data and saves it as a Parquet file."""
    if not run_task:
        current_run.log_info("Pyramid extraction skipped.")
        return

    current_run.log_info("Retrieving source DHIS2 pyramid data")

    try:
        org_units = dhis2_client.meta.organisation_units(
            fields="id,name,shortName,openingDate,closedDate,parent,level,path,geometry"
        )
        org_units = pd.DataFrame(org_units)
        org_units = org_units[org_units.level <= 5]  # filter by limit_level (fix for DRC SNIS)
        current_run.log_info(f"Organisation units extracted: {len(org_units.id.unique())}")
    except Exception as e:
        raise Exception(f"Error while extracting DHIS2 Pyramid: {e}") from e

    org_units_selection = sync_config["ORG_UNITS"].get("UIDS", [])
    if len(org_units_selection) > 0:
        org_units = select_descendants(org_units, org_units_selection)

    current_run.log_info(f"Selected organisation units: {org_units.shape[0]}.")

    # Save as Parquet
    pyramid_fname = output_dir / filename
    save_to_parquet(data=org_units, filename=pyramid_fname)
    updates_collector.setdefault("pyramid_data", []).append(pyramid_fname)
    current_run.log_info(f"DHIS2 pyramid data saved: {pyramid_fname}")


def extract_prs_data(
    pipeline_path: str,
    dhis2_client: DHIS2,
    config: dict,
    start_date: str,
    end_date: str,
    run_task: bool,
    updates_collector: dict,
    skip_existing_extracts: bool,
) -> None:
    """Resolves the extraction period and runs the data element extracts.

    Args:
        pipeline_path: Path to the pipeline directory.
        dhis2_client: DHIS2 client instance for data extraction.
        config: Configuration dictionary containing extraction settings.
        start_date: Start date for data extraction in YYYYMM format.
        end_date: End date for data extraction in YYYYMM format.
        run_task: If False, the data extraction is skipped entirely.
        updates_collector: Dictionary to collect paths of new extracts produced in this run.
        skip_existing_extracts: If True, skips downloading extracts that already exist on disk.
    """
    if not run_task:
        current_run.log_info("Data extraction skipped.")
        return

    current_run.log_info("Retrieving DHIS2 analytics data")

    # get dates and validate
    start, end = resolve_dates_and_validate(start_date, end_date, config)

    # NOTE: Adjust the start date to consider the 6 months CMM windows
    cmm_window = config["SETTINGS"].get("CMM_MONTHS_WINDOW", 6)
    start_cmm = (datetime.strptime(start, "%Y%m") - relativedelta(months=cmm_window)).strftime("%Y%m")
    extract_periods = get_extract_periods(start_cmm, end)

    # Set extractor limits
    dhis2_client.data_value_sets.MAX_DATA_ELEMENTS = 100
    dhis2_client.data_value_sets.MAX_ORG_UNITS = 100
    current_run.log_info(f"Extract periods from: {start_cmm} ({cmm_window} cmm window) to {end}")
    handle_data_element_extracts(
        pipeline_path=pipeline_path,
        dhis2_client=dhis2_client,
        config=config,
        extract_periods=extract_periods,
        updates_collector=updates_collector,
        skip_existing_extracts=skip_existing_extracts,
    )

    current_run.log_info("Extracts finished.")


def handle_data_element_extracts(
    pipeline_path: Path,
    dhis2_client: DHIS2,
    config: dict,
    extract_periods: list[str],
    updates_collector: dict,
    skip_existing_extracts: bool,
):
    """Handles data elements extracts based on the configuration."""
    data_element_extracts = config.get("DATA_ELEMENT_EXTRACTS", {}).get("EXTRACTS", [])
    if len(data_element_extracts) == 0:
        current_run.log_info("No data elements to extract.")
        return

    current_run.log_info("Starting data element extracts.")
    source_datasets = get_datasets(dhis2_client)

    if skip_existing_extracts:
        download_mode = "DOWNLOAD_NEW"
    else:
        download_mode = "DOWNLOAD_REPLACE"

    # Set extractor
    extractor = DHIS2Extractor(
        dhis2_client=dhis2_client, download_mode=download_mode, return_existing_file=skip_existing_extracts
    )
    current_run.log_info(f"Download MODE: {download_mode} - skip_existing_extracts: {skip_existing_extracts}")

    # loop over the available extract configurations
    for idx, extract in enumerate(data_element_extracts):
        extract_id = extract.get("EXTRACT_UID")
        org_units_level = extract.get("ORG_UNITS_LEVEL", None)
        data_element_uids = extract.get("UIDS", [])
        dataset_uid = extract.get("DATASET_UID")

        if extract_id is None:
            current_run.log_warning(
                f"No 'EXTRACT_UID' defined for extract position: {idx}. This is required, extract skipped."
            )
            continue

        if org_units_level is None:
            current_run.log_warning(f"No 'ORG_UNITS_LEVEL' defined for extract: {extract_id}, extract skipped.")
            continue

        if len(data_element_uids) == 0:
            current_run.log_warning(f"No data elements defined for extract: {extract_id}, extract skipped.")
            continue

        if not dataset_uid:
            current_run.log_warning(f"No dataset id defined for extract: {extract_id}, extract skipped.")
            continue

        # get org units from the dataset directly
        source_dataset = source_datasets.filter(pl.col("id").is_in([dataset_uid]))
        org_units = source_dataset["organisation_units"].explode().to_list()

        current_run.log_info(
            f"Starting data elements extract ID: '{extract_id}' ({idx + 1}) "
            f"with {len(data_element_uids)} data elements across {len(org_units)} org units "
            f"(dataset: {source_dataset['name'][0]})."
        )

        # run data elements extraction per period
        for period in extract_periods:
            try:
                extract_path = extractor.data_elements.download_period(
                    data_elements=data_element_uids,
                    org_units=org_units,
                    period=period,
                    output_dir=pipeline_path / "data" / "extracts" / "data_elements" / extract_id,
                )
                if extract_path:
                    updates_collector.setdefault(extract_id, []).append(extract_path)

            except Exception as e:
                current_run.log_error(
                    f"Extract {extract_id} download failed for period {period}, skipping to next extract. {e}"
                )
                break  # skip to next extract (if any)

        current_run.log_info(f"Extract {extract_id} finished.")


def update_dataset_with_extracts(
    pipeline_path: Path, updates_collector: dict[Path], dataset_id: str, run_task: bool
) -> None:
    """Updates the SNIS dataset with the new extracts.

    This function takes the paths of the new extracts from the updates collector and updates the OH dataset.
    NOTE: Additionally includes a json file to link the extract files with their extract_id required for integration.

    Args:
        pipeline_path: Root path of the pipeline, used to save the extract/id mapping file.
        updates_collector: Dictionary of file paths produced in this run, keyed by extract identifier.
        dataset_id: The ID of the OpenHEXA dataset to update.
        run_task: If False, the dataset update is skipped entirely.
    """
    if not run_task:
        return

    new_extracts = [item for values in updates_collector.values() for item in values]

    if not new_extracts:
        current_run.log_info("No new extracts to update in the dataset.")
        return

    mapping_file_path = pipeline_path / "data" / "updates_collector.json"
    save_updates_collector_json(updates_collector=updates_collector, output_path=mapping_file_path)

    try:
        add_files_to_dataset(
            dataset_id=dataset_id,
            file_paths=new_extracts + [mapping_file_path],
            ds_version_prefix="PRS_DS_CMM_extract",
        )
    except Exception as e:
        raise Exception(f"Error while updating SNIS dataset: {e}") from e


def save_updates_collector_json(updates_collector: dict, output_path: Path) -> None:
    """Save updates_collector as a JSON file, raising an error if it fails.

    Args:
        updates_collector: Dictionary of file paths to save, keyed by extract identifier.
        output_path: Path where the JSON mapping file will be written.

    Raises:
        RuntimeError: If the file cannot be written.
    """
    try:
        serializable = {k: [str(p.name) for p in v] for k, v in updates_collector.items()}
        with Path.open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
    except Exception as e:
        raise RuntimeError(f"Failed to save updates_collector to {output_path}: {e}") from e


if __name__ == "__main__":
    dhis2_snis_prs_cmm_morbidity_extract()
