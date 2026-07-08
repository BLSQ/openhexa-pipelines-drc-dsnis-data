from pathlib import Path

# import pandas as pd
import polars as pl
from d2d_development.extract import DHIS2Extractor
from openhexa.sdk import current_run, parameter, pipeline, workspace
from openhexa.toolbox.dhis2 import DHIS2
from openhexa.toolbox.dhis2.dataframe import get_datasets
from utils import (
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
#   -https://bluesquare.atlassian.net/browse/PATHEOC-409
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
    "add_to_dataset",
    name="Add extracts to dataset",
    help="Create a new version with the extracts created in this run.",
    type=bool,
    default=True,
    required=False,
)
def dhis2_prs_dataset_extract(start_date: str, end_date: str, run_extract_data: bool, add_to_dataset: bool):
    """Main pipeline function for DHIS2 dataset synchronization.

    Parameters
    ----------
    start_date : str
        Start date for data extraction in YYYYMM format. If not set, it will
        default to current date minus NUMBER_MONTHS_WINDOW (config).
    end_date : str
        End date for data extraction in YYYYMM format. If not set, it will default to current date minus 1.
    run_extract_data : bool, optional
        If True, runs the data extraction task (default is True).
    add_to_dataset : bool, optional
        If True, adds the extracted data to the dataset (default is True).

    Raises
    ------
    Exception
        If an error occurs during the pipeline execution.
    """
    pipeline_path = Path(workspace.files_path) / "pipelines" / "dhis2_prs_dataset_extract"
    config = load_configuration(pipeline_path / "configuration" / "extract_config.json")
    dhis2_client = connect_to_dhis2(connection_str=config["SETTINGS"]["DHIS2_CONNECTION"])
    updates_collector = {}

    try:
        # retrieve pyramid (for alignment)
        extract_pyramid(
            pipeline_path=pipeline_path,
            dhis2_snis_client=dhis2_client,
            run=run_extract_data,
            updates_collector=updates_collector,
        )

        # extract_data(
        #     pipeline_path=pipeline_path,
        #     start_date=start_date,
        #     end_date=end_date,
        #     run_task=run_extract_data,
        #     updates_collector=updates_collector,
        # )
    except Exception as e:
        current_run.log_error(f"An error occurred: {e}")
        raise

    # update_dateset(
    # add_to_dataset
    # )


def extract_pyramid(pipeline_path: str, dhis2_snis_client: DHIS2, run: bool, updates_collector: dict) -> None:
    """Pyramid extraction task.

    extracts and saves a pyramid dataframe for all levels (could be set via config in the future)
    """
    if not run:
        current_run.log_info("Pyramid extraction skipped.")
        return
    current_run.log_info("Retrieving SNIS DHIS2 pyramid data")

    try:
        # retrieve full pyramid
        org_units = dhis2_snis_client.meta.organisation_units(
            fields="id,name,shortName,openingDate,closedDate,parent,level,path,geometry"
        )
    except Exception as e:
        raise Exception(f"Error while retrieving SNIS DHIS2 Pyramid: {e}") from e

    org_units = pl.DataFrame(org_units)
    org_units = org_units.filter(pl.col("level") <= 5)  # Select up to level 5
    org_units = org_units.sort("level", descending=False)
    current_run.log_info(f"{org_units['id'].n_unique()} units at organisation unit level '5'")

    # Save as Parquet
    pyramid_path = pipeline_path / "data" / "pyramid"
    save_to_parquet(data=org_units, filename=pyramid_path / "snis_pyramid.parquet")
    current_run.log_info(f"SNIS DHIS2 pyramid data saved: {pyramid_path / 'snis_pyramid.parquet'}")

    # add to updates collector
    updates_collector.setdefault("pyramid", []).append(pyramid_path / "snis_pyramid.parquet")


# this logic should be in the integration side
# def pyramid_selection_for_prs(
#     pyramid: pl.DataFrame,
#     org_units_selection: list[str],
#     include_children: bool = True,
# ) -> pl.DataFrame:
#     """Filter and select organisation units from the DHIS2 pyramid for the PRS extract.

#     -Keeps all descendants of the org_units_selection parents.
#     -If include_children is False, only the selected organisation units themselves are kept.

#     Args:
#         pyramid: Full organisation unit pyramid data.
#         org_units_selection: List of organisation unit ids to select. If empty, no selection filtering is applied.
#         include_children: If True, includes all descendants of the selected organisation units. If False, only the
#             selected organisation units themselves are kept.

#     Returns:
#         The filtered organisation unit pyramid.
#     """
#     org_units = pyramid.clone()
#     if org_units_selection:
#         if include_children:
#             org_units = select_descendants(org_units, org_units_selection)
#         else:
#             org_units = org_units.filter(pl.col("id").is_in(org_units_selection))

#     current_run.log_info(f"Selected organisation units: {org_units['id'].n_unique()}.")
#     return org_units


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

    config = load_configuration(pipeline_path / "configuration" / "extract_config.json")
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


if __name__ == "__main__":
    dhis2_prs_dataset_extract()
