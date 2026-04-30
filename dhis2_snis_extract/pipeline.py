from itertools import product
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


@pipeline("dhis2_snis_extract")
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
    name="Extract Organisation Units",
    help="",
    type=bool,
    default=True,
    required=False,
)
@parameter(
    "run_pop",
    name="Extrat population",
    help="",
    type=bool,
    default=True,
    required=False,
)
@parameter(
    "run_analytics",
    name="Extract analytics",
    help="",
    type=bool,
    default=True,
    required=False,
)
@parameter(
    "add_to_dataset",
    name="Add extracts to dataset",
    help="Create a new version with the extracts created in this run.",
    type=bool,
    default=True,
    required=False,
)
def dhis2_snis_extract(
    run_orgunits: bool, run_pop: bool, run_analytics: bool, start_date: str, end_date: str, add_to_dataset: bool
) -> None:
    """Simple pipeline to retrieve the a monthly PNLP related data extract from DHIS2 SNIS instance."""
    pipeline_path = Path(workspace.files_path) / "pipelines" / "dhis2_snis_extract"

    try:
        config = load_configuration(pipeline_path / "config" / "snis_extraction_config.json")
        dhis2_client = connect_to_dhis2(connection_str=config["SETTINGS"]["DHIS2_CONNECTION"])
        updates_collector = {}

        # retrieve pyramid (for alignment)
        extract_pyramid(
            pipeline_path=pipeline_path,
            dhis2_snis_client=dhis2_client,
            run=run_orgunits,
            updates_collector=updates_collector,
        )

        extract_population(
            pipeline_path=pipeline_path,
            dhis2_snis_client=dhis2_client,
            start_date=start_date,
            end_date=end_date,
            config=config,
            run=run_pop,
            updates_collector=updates_collector,
        )

        extract_analytics(
            pipeline_path=pipeline_path,
            start_date=start_date,
            end_date=end_date,
            config=config,
            dhis2_snis_client=dhis2_client,
            run=run_analytics,
            updates_collector=updates_collector,
        )

        update_snis_dataset(
            updates_collector=updates_collector,
            dataset_id="snis-extracts",
            run=add_to_dataset,
        )

        current_run.log_info("Pipeline execution completed successfully.")

    except Exception as e:
        current_run.log_error(f"Error occurred: {e}")


def extract_pyramid(pipeline_path: str, dhis2_snis_client: DHIS2, run: bool, updates_collector: dict) -> None:
    """Pyramid extraction task.

    extracts and saves a pyramid dataframe for all levels (could be set via config in the future)
    """
    if not run:
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


def extract_population(
    pipeline_path: str,
    start_date: str,
    end_date: str,
    dhis2_snis_client: str,
    config: dict,
    run: bool,
    updates_collector: dict,
) -> None:
    """Population data extraction task."""
    # NOTE: Population data is extracted per year period. Is enough to extract population once per year,
    # but we need to account for updates at the beginning of each year, so we keep retrieving and pushing.
    if not run:
        return True

    current_run.log_info("Retrieving SNIS DHIS2 population data")

    # get dates and validate  # TO BE DELETED
    start, end = resolve_dates_and_validate(start_date, end_date, config)
    extract_periods = [p[:4] for p in get_extract_periods(start, end)]
    extract_periods = list(set(extract_periods))
    extract_periods.sort()

    # retrieve AIRE de sante list from SNIS
    aires_list = get_ou_list(pyramid_fname=pipeline_path / "data" / "pyramid" / "snis_pyramid.parquet", ou_level=4)

    # set folder for population extracts
    pop_path = pipeline_path / "data" / "population"
    try:
        for period in extract_periods:
            # retrieve
            raw_pop_data = dhis2_snis_client.data_value_sets.get(
                data_elements=config["POPULATION_UIDS"],
                org_units=aires_list,
                periods=[period],
            )

            population_table = pd.DataFrame(raw_pop_data)
            population_table_formatted = map_to_snis_format(dhis_data=population_table, data_type="POPULATION")

            # Save as Parquet
            pop_file = pop_path / f"snis_population_{period}.parquet"
            if pop_file.exists():
                current_run.log_info(f"Replacing population extract for period {period}.")
            else:
                current_run.log_info(f"Saving population extract for period {period}.")

            save_to_parquet(population_table_formatted, filename=pop_file)

            # add to updates collector
            updates_collector.setdefault("population", []).append(pop_file)

    except Exception as e:
        raise Exception(f"Population task error : {e}") from e


def extract_analytics(
    pipeline_path: str,
    start_date: str,
    end_date: str,
    config: dict,
    dhis2_snis_client: DHIS2,
    run: bool,
    updates_collector: dict,
) -> None:
    """Data extraction task."""
    if not run:
        return

    current_run.log_info("Retrieving SNIS DHIS2 analytics data")

    # get dates and validate
    start, end = resolve_dates_and_validate(start_date, end_date, config)
    extract_periods = get_extract_periods(start, end)

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
            current_run.log_info(f"Retrieving SNIS data extract for period : {period}")
            handle_extract_for_period(pipeline_path, dhis2_snis_client, period, fosa_list, config, updates_collector)

        current_run.log_info("Process finished.")
    except Exception as e:
        raise Exception(f"Extract task error : {e}") from e


def handle_extract_for_period(
    pipeline_path: Path, dhis2_client: DHIS2, period: str, org_unit_list: list, config: dict, updates_collector: dict
) -> None:
    """Wrapper function to handle the data extraction for a given period, with error handling and logging."""
    # Setup extractor
    dhis2_extractor = DHIS2Extractor(dhis2_client=dhis2_client, download_mode=config["SETTINGS"]["MODE"])

    raw_routine_path = retrieve_snis_routine_extract(
        extractor=dhis2_extractor,
        data_element_uids=config["ROUTINE_DATA_ELEMENT_UIDS"],
        period=period,
        org_unit_list=org_unit_list,
        file_path=pipeline_path / "data" / "routine",
    )

    raw_rates_path = retrieve_snis_rates_extract(
        extractor=dhis2_extractor,
        rate_ids=config["RATE_UIDS"],
        period=period,
        org_unit_list=org_unit_list,
        file_path=pipeline_path / "data" / "reporting_rates",
    )
    raw_acm_path = retrieve_snis_acm_extract(
        extractor=dhis2_extractor,
        acm_indicator_ids=config["ACM_INDICATOR_UIDS"],
        period=period,
        org_unit_list=org_unit_list,
        file_path=pipeline_path / "data" / "indicators",
    )

    # Build snis extract (merge for compatibility with old format)
    build_snis_extract(
        raw_routine_path,
        raw_rates_path,
        raw_acm_path,
        period,
        pipeline_path / "data" / "snis_extracts" / f"snis_data_{period}.parquet",
        updates_collector=updates_collector,
    )


def retrieve_snis_routine_extract(
    extractor: DHIS2Extractor,
    data_element_uids: list,
    period: str,
    org_unit_list: list,
    file_path: Path,
) -> Path:
    """Retrieves routine data from DHIS2 for the specified data elements, period, and organizational units.

    Returns
    -------
        Path
            The file path where the extracted data is saved in Parquet format.
    """
    return extractor.data_elements.download_period(
        data_elements=data_element_uids,
        org_units=org_unit_list,
        period=period,
        output_dir=file_path,
    )


def retrieve_snis_rates_extract(
    extractor: DHIS2Extractor, rate_ids: dict, period: str, org_unit_list: list, file_path: Path
) -> Path:
    """Retrieves rates data from DHIS2.

    Returns
    -------
        Path
            The file path where the extracted data is saved in Parquet format.
    """
    if period >= "202501":
        rate_year = rate_ids.get("2025")
    else:
        rate_year = rate_ids.get("2024")
    reporting_datasets = rate_year.get("DATASETS", [])
    reporting_metrics = rate_year.get("METRICS", {}).keys()
    reporting_combinations = [f"{ds}.{metric}" for ds, metric in product(reporting_datasets, reporting_metrics)]
    current_run.log_debug(f"Reporting ids: {reporting_combinations}")

    return extractor.reporting_rates.download_period(
        reporting_rates=reporting_combinations, org_units=org_unit_list, period=period, output_dir=file_path
    )


def retrieve_snis_acm_extract(
    extractor: DHIS2Extractor, acm_indicator_ids: list, period: str, org_unit_list: list, file_path: Path
) -> Path:
    """Retrieves ACM indicator data from DHIS2 for the specified period and organizational units.

    Returns
    -------
        Path
            The file path where the extracted data is saved in Parquet format.
    """
    return extractor.indicators.download_period(
        indicators=acm_indicator_ids, period=period, org_units=org_unit_list, output_dir=file_path
    )


def build_snis_extract(
    raw_routine_path: Path,
    raw_rates_path: Path,
    raw_acm_path: Path,
    period: str,
    file_path: Path,
    updates_collector: dict,
) -> None:
    """Builds the SNIS data extract by merging routine, rates, and ACM data.

    This function reads the raw routine, rates, and ACM data from their respective file paths,
    merges them into a single DataFrame, and saves the merged extract in Parquet format.
    """
    # read raw data
    routine_df = pd.read_parquet(raw_routine_path)
    rates_df = pd.read_parquet(raw_rates_path)
    acm_df = pd.read_parquet(raw_acm_path)

    # merge all extracts (union)
    merged_extract = pd.concat([routine_df, rates_df, acm_df], ignore_index=True)
    if file_path.exists():
        current_run.log_info(f"Replacing extract for period {period}.")
    else:
        current_run.log_info(f"Saving extract for period {period}.")
    save_to_parquet(merged_extract, file_path)
    updates_collector.setdefault("snis_extracts", []).append(file_path)


def update_snis_dataset(updates_collector: dict[Path], dataset_id: str, run: bool) -> None:
    """Updates the SNIS dataset with the new extracts.

    This function takes the paths of the new extracts from the updates collector and updates the OH dataset.
    """
    if not run:
        return

    new_extracts = [item for values in updates_collector.values() for item in values]

    if not new_extracts:
        current_run.log_info("No new extracts to update in the dataset.")
        return

    try:
        add_files_to_dataset(
            dataset_id=dataset_id,
            file_paths=new_extracts,
            ds_version_prefix="SNIS_extract",
        )
    except Exception as e:
        raise Exception(f"Error while updating SNIS dataset: {e}") from e


def map_to_snis_format(
    dhis_data: pd.DataFrame,
    data_type: str = "DATAELEMENT",
    domain_type: str = "AGGREGATED",
) -> pd.DataFrame:
    """Maps DHIS2 data to a standardized data extraction table.

    Parameters
    ----------
    dhis_data : pd.DataFrame
        Input DataFrame containing DHIS2 data. Must include columns like `period`, `orgUnit`,
        `categoryOptionCombo(DATAELEMENT)`, `attributeOptionCombo(DATAELEMENT)`, `dataElement`
        and `value` based on the data type.
    data_type : str
        The type of data being mapped. Supported values are:
        - "DATAELEMENT": Includes `categoryOptionCombo` and maps `dataElement` to `dx_uid`.
        - "DATASET": Maps `dx` to `dx_uid` and `rate_type` by split the string by `.`.
        - "INDICATOR": Maps `dx` to `dx_uid`.
        - "POPULATION": Maps `dx` to `dx_uid` and the rest of DHIS2 raw columns
        Default is "DATAELEMENT".
    domain_type : str, optional
        The domain of the data if its per period (Agg ex: monthly) or datapoint (Tracker ex: per day):
        - "AGGREGATED": For aggregated data (default).
        - "TRACKER": For tracker data.

    Returns
    -------
    pd.DataFrame
        A DataFrame formatted to SNIS standards, with the following columns:
        - "data_type": The type of data (DATAELEMENT, DATASET, or INDICATOR).
        - "dx_uid": Data element, dataset, or indicator UID.
        - "period": Reporting period.
        - "orgUnit": Organization unit.
        - "categoryOptionCombo": (Only for DATAELEMENT) Category option combo UID.
        - "rate_type": (Only for DATASET) Rate type.
        - "domain_type": Data domain (AGGREGATED or TRACKER).
        - "value": Data value.
    """
    if dhis_data.empty:
        return None

    if data_type not in ["DATAELEMENT", "DATASET", "INDICATOR", "POPULATION"]:
        raise ValueError("Incorrect 'data_type' configuration ('DATAELEMENT', 'DATASET', 'INDICATOR', 'POPULATION')")

    try:
        snis_format = pd.DataFrame(
            columns=[
                "data_type",
                "dx_uid",
                "period",
                "org_unit",
                "category_option_combo",
                "attribute_option_combo",
                "rate_type",
                "domain_type",
                "value",
            ]
        )
        snis_format["period"] = dhis_data.period
        snis_format["org_unit"] = dhis_data.orgUnit
        snis_format["domain_type"] = domain_type
        snis_format["value"] = dhis_data.value
        snis_format["data_type"] = data_type
        if data_type in ["DATAELEMENT", "POPULATION"]:
            snis_format["dx_uid"] = dhis_data.dataElement
            snis_format["category_option_combo"] = dhis_data.categoryOptionCombo
            snis_format["attribute_option_combo"] = dhis_data.attributeOptionCombo
        if data_type == "DATASET":
            snis_format[["dx_uid", "rate_type"]] = dhis_data.dx.str.split(".", expand=True)
        if data_type == "INDICATOR":
            snis_format["dx_uid"] = dhis_data.dx
        return snis_format
    except Exception as e:
        raise Exception(f"Unexpected Error while creating routine format table: {e}") from e


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
    dhis2_snis_extract()
