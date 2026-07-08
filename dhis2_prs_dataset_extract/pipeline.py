import json
from pathlib import Path

import polars as pl
from d2d_development.extract import DHIS2Extractor
from openhexa.sdk import current_run, parameter, pipeline, workspace
from openhexa.toolbox.dhis2 import DHIS2
from utils import (
    add_files_to_dataset,
    connect_to_dhis2,
    get_extract_periods,
    load_configuration,
    read_parquet_extract,
    resolve_dates_and_validate,
    save_to_parquet,
    select_descendants,
)

# Ticket(s) related to this pipeline:
#   -https://bluesquare.atlassian.net/browse/SANRUSSC24-32
#   -https://bluesquare.atlassian.net/browse/SAN-122
#   -https://bluesquare.atlassian.net/browse/SAN-125
#   -https://bluesquare.atlassian.net/browse/PATHEOC-409 (latest)
# github repo:
#   -https://github.com/BLSQ/openhexa-pipelines-drc-prs


@pipeline("dhis2_prs_dataset_extract", timeout=43200)  # 3600 * 12 hours
@parameter(
    code="start_date",
    name="Start date (format: YYYYMM)",
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
    type=str,
    required=False,
    help=(
        "End date for data extraction in YYYYMM format. "
        "If not set, it will default to current date minus NUMBER_MONTHS_WINDOW."
    ),
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
def dhis2_prs_dataset_extract(
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
    pipeline_path = Path(workspace.files_path) / "pipelines" / "dhis2_prs_dataset_extract"
    config = load_configuration(pipeline_path / "configuration" / "extract_config.json")
    dhis2_client = connect_to_dhis2(connection_str=config["SETTINGS"]["DHIS2_CONNECTION"])
    updates_collector = {}

    try:
        # retrieve PRS (selection) pyramid (for alignment)
        extract_pyramid_for_prs(
            pipeline_path=pipeline_path,
            dhis2_snis_client=dhis2_client,
            run=run_extract_data,
            updates_collector=updates_collector,
            skip_existing_extracts=skip_existing_extracts,
        )

        extract_prs_data(
            pipeline_path=pipeline_path,
            dhis2_client=dhis2_client,
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
            dataset_id="snis-prs-dataset-sync",
            run_task=add_to_dataset,
        )
    except Exception as e:
        current_run.log_error(f"An error occurred while updating the dataset: {e}")
        raise


def extract_pyramid_for_prs(
    pipeline_path: str, dhis2_snis_client: DHIS2, run: bool, updates_collector: dict, skip_existing_extracts: bool
) -> None:
    """Extracts and saves the org unit pyramid for the PRS extract.

    Args:
        pipeline_path: Root path of the pipeline, used to load configuration and save output files.
        dhis2_snis_client: An instance of the DHIS2 client used to retrieve the pyramid.
        run: If False, the pyramid extraction is skipped entirely.
        updates_collector: Dictionary accumulating the paths of files produced in this run, keyed
            by extract identifier. Updated in place with the pyramid file path.
        skip_existing_extracts: If True and the pyramid file already exists on disk, reuses it
            instead of retrieving a new one from DHIS2.
    """
    if not run:
        current_run.log_info("Pyramid extraction skipped.")
        return

    # path
    pyramid_path = pipeline_path / "data" / "pyramid" / "snis_prs_pyramid.parquet"
    if skip_existing_extracts and pyramid_path.exists():
        current_run.log_info(f"Pyramid extraction skipped, file already exists: {pyramid_path}")
        updates_collector.setdefault("pyramid", []).append(pyramid_path)
        return

    current_run.log_info("Retrieving SNIS DHIS2 pyramid data")
    config_sync = load_configuration(pipeline_path / "configuration" / "sync_config.json")
    limit_lvl = config_sync.get("ORG_UNITS", {}).get("SELECTION", {}).get("LIMIT_LEVEL", [])

    try:
        # retrieve full pyramid
        org_units = dhis2_snis_client.meta.organisation_units(
            fields="id,name,shortName,openingDate,closedDate,parent,level,path,geometry"
        )
    except Exception as e:
        raise Exception(f"Error while retrieving SNIS DHIS2 Pyramid: {e}") from e

    org_units = pl.DataFrame(org_units)
    org_units = org_units.filter(pl.col("level") <= limit_lvl)  # Select up to level 5 (clean)
    org_units = org_units.sort("level", descending=False)
    current_run.log_info(f"{org_units['id'].n_unique()} units at organisation unit level '{limit_lvl}'")

    # NOTE: FOR PRS integration we only focus on 20 provinces (see: sync_config.json).
    org_units = pyramid_selection_for_prs(
        pyramid=org_units,
        org_units_selection=config_sync.get("ORG_UNITS", {}).get("SELECTION", {}).get("UIDS", []),
        include_children=config_sync.get("ORG_UNITS", {}).get("SELECTION", {}).get("INCLUDE_CHILDREN", []),
    )

    # Save as Parquet
    save_to_parquet(data=org_units, filename=pyramid_path)
    current_run.log_info(f"SNIS DHIS2 pyramid data saved: {pyramid_path}")

    # add to updates collector
    updates_collector.setdefault("pyramid", []).append(pyramid_path)


def pyramid_selection_for_prs(
    pyramid: pl.DataFrame,
    org_units_selection: list[str],
    include_children: bool = True,
) -> pl.DataFrame:
    """Filter and select organisation units from the DHIS2 pyramid for the PRS extract.

    -Keeps all descendants of the org_units_selection parents.
    -If include_children is False, only the selected organisation units themselves are kept.

    Args:
        pyramid: Full organisation unit pyramid data.
        org_units_selection: List of organisation unit ids to select. If empty, no selection filtering is applied.
        include_children: If True, includes all descendants of the selected organisation units. If False, only the
            selected organisation units themselves are kept.

    Returns:
        The filtered organisation unit pyramid.
    """
    org_units = pyramid.clone()
    if org_units_selection:
        if include_children:
            org_units = select_descendants(org_units, org_units_selection)
        else:
            org_units = org_units.filter(pl.col("id").is_in(org_units_selection))

    current_run.log_info(f"Selected organisation units: {org_units['id'].n_unique()}.")
    return org_units


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
        pipeline_path: Root path of the pipeline, used to locate configuration and output files.
        dhis2_client: An instance of the DHIS2 client used to retrieve data.
        config: Pipeline configuration dictionary (extract_config.json).
        start_date: Start date for data extraction in YYYYMM format.
        end_date: End date for data extraction in YYYYMM format.
        run_task: If False, the data extraction is skipped entirely.
        updates_collector: Dictionary accumulating the paths of files produced in this run, keyed
            by extract identifier. Updated in place by the downstream extract handlers.
        skip_existing_extracts: If True, skips downloading extracts that already exist on disk,
            reusing the existing files instead.
    """
    if not run_task:
        current_run.log_info("Data extraction skipped.")
        return

    current_run.log_info("Retrieving DHIS2 analytics data")

    # get dates and validate
    start, end = resolve_dates_and_validate(start_date, end_date, config)
    extract_periods = get_extract_periods(start, end)
    current_run.log_info(f"Download from: {start} to {end}")

    # limits
    dhis2_client.analytics.MAX_DX = 100
    dhis2_client.analytics.MAX_ORG_UNITS = 100
    dhis2_client.data_value_sets.MAX_DATA_ELEMENTS = 100
    dhis2_client.data_value_sets.MAX_ORG_UNITS = 100

    handle_data_element_extracts(
        pipeline_path=pipeline_path,
        dhis2_client=dhis2_client,
        config=config,
        pyramid=read_parquet_extract(pipeline_path / "data" / "pyramid" / "snis_prs_pyramid.parquet"),
        extract_periods=extract_periods,
        updates_collector=updates_collector,
        skip_existing_extracts=skip_existing_extracts,
    )

    current_run.log_info("Extracts finished.")


def handle_data_element_extracts(
    pipeline_path: Path,
    dhis2_client: DHIS2,
    config: dict,
    pyramid: pl.DataFrame,
    extract_periods: list,
    updates_collector: dict[Path],
    skip_existing_extracts: bool,
) -> None:
    """Handles data elements extracts based on the configuration.

    Args:
        pipeline_path: Root path of the pipeline, used to build each extract's output directory.
        dhis2_client: An instance of the DHIS2 client used to retrieve data.
        config: Pipeline configuration dictionary, expected to contain a "DATA_ELEMENTS" section.
        pyramid: Org unit pyramid used to resolve the org units for each extract's configured level.
        extract_periods: List of periods (YYYYMM) to extract data for.
        updates_collector: Dictionary accumulating the paths of files produced in this run, keyed
            by extract identifier. Updated in place with each successfully downloaded file.
        skip_existing_extracts: If True, skips downloading extracts that already exist on disk,
            reusing the existing files instead.
    """
    data_element_extracts = config.get("DATA_ELEMENTS", {}).get("EXTRACTS", [])

    if not data_element_extracts:
        current_run.log_info("No data element to extract.")
        return
    current_run.log_info("Starting data element extracts.")

    if skip_existing_extracts:
        download_mode = "DOWNLOAD_NEW"
    else:
        download_mode = config.get("SETTINGS", {}).get("MODE", "DOWNLOAD_REPLACE")
    current_run.log_info(f"Download mode: {download_mode}")

    dhis2_extractor = DHIS2Extractor(
        dhis2_client=dhis2_client, download_mode=download_mode, return_existing_file=skip_existing_extracts
    )
    # loop over the available extract configurations
    for idx, extract in enumerate(data_element_extracts):
        extract_id = extract.get("EXTRACT_UID")
        ou_level = extract.get("ORG_UNITS_LEVEL")
        data_element_uids = extract.get("UIDS", [])

        if not extract_id:
            current_run.log_warning(f"No 'EXTRACT_UID' defined at position: {idx}. This is required, extract skipped.")
            continue

        if not ou_level:
            current_run.log_warning(f"No 'ORG_UNITS_LEVEL' defined for extract: {extract_id}, extract skipped.")
            continue

        if not data_element_uids:
            current_run.log_warning(f"No 'UIDS' data elements defined for extract: {extract_id}, extract skipped.")
            continue

        org_units = pyramid.filter(pl.col("level") == ou_level)["id"].to_list()
        current_run.log_info(
            f"Starting data elements extract ID: '{extract_id}' ({idx + 1}) "
            f"with {len(data_element_uids)} data elements across {len(org_units)} org units."
        )

        for period in extract_periods:
            try:
                extract_path = dhis2_extractor.data_elements.download_period(
                    data_elements=data_element_uids,
                    org_units=org_units,
                    period=period,
                    output_dir=pipeline_path / "data" / "data_elements" / f"extract_{extract_id}",
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
            ds_version_prefix="PRS_DS_SYNC_extract",
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
    dhis2_prs_dataset_extract()
