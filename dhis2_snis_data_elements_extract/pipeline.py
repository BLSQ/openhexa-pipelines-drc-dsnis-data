from pathlib import Path

import pandas as pd
from d2d_development.extract import DHIS2Extractor
from openhexa.sdk import current_run, parameter, pipeline, workspace
from openhexa.toolbox.dhis2 import DHIS2
from utils import (
    add_files_to_dataset,
    connect_to_dhis2,
    get_extract_periods,
    load_configuration,
    resolve_dates_and_validate,
    save_to_parquet,
)


@pipeline("dhis2_snis_data_elements_extract", timeout=28800)
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
    "run_orgunits",
    name="Extract org units",
    type=bool,
    default=True,
    help="Run organisation units extract.",
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
def dhis2_snis_data_elements_extract(
    start_date: str, end_date: str, run_orgunits: bool, run_extract_data: bool, add_to_dataset: bool
):
    """Write your pipeline orchestration here.

    Pipeline functions should only call tasks and should never perform IO operations or expensive computations.
    """
    pipeline_path = Path(workspace.files_path) / "pipelines" / "dhis2_snis_data_elements_extract"
    updates_collector = {}

    try:
        config = load_configuration(pipeline_path / "config" / "extract_config.json")
        dhis2_client = connect_to_dhis2(connection_str=config["SETTINGS"]["DHIS2_CONNECTION"])
        updates_collector = {}

        extract_pyramid(
            pipeline_path=pipeline_path,
            dhis2_snis_client=dhis2_client,
            run_task=run_orgunits,
            updates_collector=updates_collector,
        )

        extract_data(
            pipeline_path=pipeline_path,
            start_date=start_date,
            end_date=end_date,
            config=config,
            dhis2_snis_client=dhis2_client,
            run_task=run_extract_data,
            updates_collector=updates_collector,
        )

        update_snis_dataset(
            updates_collector=updates_collector,
            dataset_id="snis-data-elements-extracts",
            run_task=add_to_dataset,
        )

        current_run.log_info("Pipeline execution completed successfully.")

    except Exception as e:
        current_run.log_error(f"An error occurred: {e}")
        raise


def extract_pyramid(pipeline_path: str, dhis2_snis_client: DHIS2, run_task: bool, updates_collector: dict) -> None:
    """Pyramid extraction task.

    extracts and saves a pyramid dataframe for all levels (could be set via config in the future)
    """
    if not run_task:
        return

    current_run.log_info("Retrieving SNIS DHIS2 pyramid data")

    try:
        # retrieve full pyramid
        org_units = dhis2_snis_client.meta.organisation_units(
            fields="id,name,shortName,openingDate,closedDate,parent,level,path,geometry"
        )
        org_units = pd.DataFrame(org_units)
        org_units = org_units[org_units.level <= 5]  # Select level 5
        current_run.log_info(
            f"Extracted {len(org_units[org_units.level == 5].id.unique())} units at organisation unit level {5}"
        )

        # Save as Parquet
        pyramid_path = pipeline_path / "data" / "pyramid"
        save_to_parquet(data=org_units, filename=pyramid_path / "snis_pyramid.parquet")
        current_run.log_info(f"SNIS DHIS2 pyramid data saved: {pyramid_path / 'snis_pyramid.parquet'}")

        # add to updates collector
        updates_collector.setdefault("pyramid", []).append(pyramid_path / "snis_pyramid.parquet")

    except Exception as e:
        raise Exception(f"Error while extracting SNIS DHIS2 Pyramid: {e}") from e


def extract_data(
    pipeline_path: str,
    start_date: str,
    end_date: str,
    config: dict,
    dhis2_snis_client: DHIS2,
    run_task: bool,
    updates_collector: dict,
) -> None:
    """Data extraction task."""
    if not run_task:
        return

    current_run.log_info("Retrieving DHIS2 analytics data")

    # get dates and validate
    start, end = resolve_dates_and_validate(start_date, end_date, config)
    extract_periods = get_extract_periods(start, end)

    if start < "202501" or end < "202501":
        current_run.log_error("Invalid date range: periods before January 2025 are not allowed.")
        raise ValueError

    # retrieve FOSA ids from SNIS
    fosa_list = get_ou_list(pyramid_fname=pipeline_path / "data" / "pyramid" / "snis_pyramid.parquet", ou_level=5)
    current_run.log_info(f"Download MODE: {config['SETTINGS']['MODE']} from: {start} to {end}")

    # limits
    dhis2_snis_client.analytics.MAX_DX = 100
    dhis2_snis_client.analytics.MAX_ORG_UNITS = 100
    dhis2_snis_client.data_value_sets.MAX_DATA_ELEMENTS = 100
    dhis2_snis_client.data_value_sets.MAX_ORG_UNITS = 100

    try:
        for period in extract_periods:
            handle_extract_for_period(
                pipeline_path=pipeline_path,
                dhis2_client=dhis2_snis_client,
                period=period,
                org_unit_list=fosa_list,
                config=config,
                updates_collector=updates_collector,
            )

        current_run.log_info("Process finished.")
    except Exception as e:
        raise Exception(f"Extract task error : {e}") from e


def handle_extract_for_period(
    pipeline_path: Path, dhis2_client: DHIS2, period: str, org_unit_list: list, config: dict, updates_collector: dict
) -> None:
    """Wrapper function to handle the data extraction for a given period, with error handling and logging."""
    # Setup extractor
    dhis2_extractor = DHIS2Extractor(dhis2_client=dhis2_client, download_mode=config["SETTINGS"]["MODE"])

    raw_data_path = dhis2_extractor.data_elements.download_period(
        data_elements=config["DATA_ELEMENTS"]["UIDS"],
        org_units=org_unit_list,
        period=period,
        output_dir=pipeline_path / "data" / "extracts",
        filename=f"data_{period}.parquet",
    )
    if raw_data_path:
        updates_collector.setdefault("snis_extracts", []).append(raw_data_path)
    else:
        current_run.log_info(f"No data added for period {period}.")


def update_snis_dataset(updates_collector: dict[Path], dataset_id: str, run_task: bool) -> None:
    """Updates the SNIS dataset with the new extracts.

    This function takes the paths of the new extracts from the updates collector and updates the OH dataset.
    """
    if not run_task:
        return

    new_extracts = [item for values in updates_collector.values() for item in values]

    if not new_extracts:
        current_run.log_info("No new extracts to update in the dataset.")
        return

    try:
        add_files_to_dataset(
            dataset_id=dataset_id,
            file_paths=new_extracts,
            ds_version_prefix="SNIS_DE_extract",
        )
    except Exception as e:
        raise Exception(f"Error while updating SNIS dataset: {e}") from e


def get_ou_list(pyramid_fname: Path, ou_level: int) -> list:
    """Retrieves a list of organizational unit IDs from the pyramid Parquet file based on the specified OU level.

    Returns
    -------
    list
        A list of organizational unit IDs corresponding to the specified OU level.
    """
    try:
        # Retrieve organisational units and filter by ou_level
        ous = pd.read_parquet(pyramid_fname)
        ou_list = ous.loc[ous.level == ou_level].id.to_list()
    except Exception as e:
        raise Exception(f"Error loading pyramid file: {e}") from e

    current_run.log_info(f"DHIS2 org units id list {len(ou_list)} at level {ou_level}")
    return ou_list


if __name__ == "__main__":
    dhis2_snis_data_elements_extract()
