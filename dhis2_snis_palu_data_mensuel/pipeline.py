from itertools import product
from pathlib import Path

import polars as pl
from d2d_development.extract import DHIS2Extractor
from openhexa.sdk import current_run, parameter, pipeline, workspace
from openhexa.toolbox.dhis2 import DHIS2
from openhexa.toolbox.dhis2.dataframe import get_organisation_units
from utils import (
    add_files_to_dataset,
    connect_to_dhis2,
    get_extract_periods,
    load_configuration,
    resolve_dates_and_validate,
    save_to_parquet,
)


@pipeline("dhis2_snis_palu_data_mensuel", timeout=21600)  # 6 hours
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
    help="Extract data elements from SNIS.",
)
@parameter(
    code="add_to_dataset",
    name="Add extracted data to dataset.",
    type=bool,
    default=True,
    help="Add extracts created in this run to the  dataset.",
)
def dhis2_snis_palu_data_mensuel(start_date: str, end_date: str, run_extract_data: bool, add_to_dataset: bool):
    """Orchestrates the SNIS palu monthly extraction, compilation, and dataset update.

    Args:
        start_date (str): Start date for data extraction in YYYYMM format.
        end_date (str): End date for data extraction in YYYYMM format.
        run_extract_data (bool): Whether to run the DHIS2 data extraction step.
        add_to_dataset (bool): Whether to push the compiled extracts to the OpenHEXA dataset.
    """
    pipelines_root = Path(workspace.files_path) / "pipelines"
    pipeline_path = pipelines_root / "dhis2_snis_palu_data_mensuel"

    try:
        config = load_configuration(pipeline_path / "config" / "extract_config.json")
        dhis2_client = connect_to_dhis2(connection_str=config["SETTINGS"]["DHIS2_CONNECTION"])
        extract_periods = resolve_extract_periods(start_date, end_date, config)
    except Exception as e:
        current_run.log_error(f"Error during setup: {e}")
        raise

    if extract_periods[0] < "202501" or extract_periods[-1] < "202501":
        msg = "Invalid date range: periods before January 2025 are not allowed."
        current_run.log_error(msg)
        raise ValueError(msg)

    try:
        extract_pyramid_metadata(pipeline_path=pipeline_path, dhis2_snis_client=dhis2_client, run_task=run_extract_data)

        extract_data(
            pipeline_path=pipeline_path,
            extract_periods=extract_periods,
            config=config,
            dhis2_snis_client=dhis2_client,
            run_task=run_extract_data,
        )

        current_run.log_info("Data extracted successfully.")

    except Exception as e:
        current_run.log_error(f"An error occurred: {e}")
        raise

    try:
        palu_extract_paths = compile_palu_extracts(
            extract_periods=extract_periods,
            data_path=pipeline_path / "data",
            snis_extracts_path=pipelines_root / "dhis2_snis_extract" / "data",
            output_path=pipeline_path / "data" / "palu_extracts",
            config_path=pipeline_path / "config",
            run_task=run_extract_data,
        )
    except Exception as e:
        current_run.log_error(f"An error while compiling data: {e}")
        raise

    try:
        update_snis_dataset(
            pipeline_path=pipeline_path,
            snis_extracts_path=pipelines_root / "dhis2_snis_extract" / "data",
            new_extracts=palu_extract_paths,
            extract_periods=extract_periods,
            dataset_id="snis-palu-mensuel-extracts",
            run_task=add_to_dataset,
        )
        current_run.log_info("Dataset updated successfully.")
    except Exception as e:
        current_run.log_error(f"An error occurred while updating the dataset: {e}")
        raise


def extract_pyramid_metadata(pipeline_path: str, dhis2_snis_client: DHIS2, run_task: bool) -> None:
    """Extracts and saves the pyramid metadata at level 5.

    Args:
        pipeline_path (str): Root path of the pipeline used to resolve the output data folder.
        dhis2_snis_client (DHIS2): Connected DHIS2 client used to retrieve the pyramid.
        run_task (bool): Whether to run this extraction step.
    """
    if not run_task:
        current_run.log_info("Skipping pyramid metadata extraction as run_task is set to False.")
        return

    current_run.log_info("Retrieving SNIS DHIS2 pyramid metadata")

    try:
        # retrieve full pyramid
        org_units = get_organisation_units(dhis2_snis_client).drop("geometry")
        org_units = org_units.filter(pl.col("level") == 5)
        current_run.log_info(f"{len(org_units['id'].unique())} units at organisation unit level: 5")
    except Exception as e:
        raise Exception(f"Error while extracting SNIS DHIS2 Pyramid: {e}") from e

    # Save as Parquet
    pyramid_path = pipeline_path / "data" / "pyramid_metadata"
    save_to_parquet(data=org_units, filename=pyramid_path / "snis_pyramid_metadata.parquet")
    current_run.log_info(f"SNIS DHIS2 pyramid metadata saved: {pyramid_path / 'snis_pyramid_metadata.parquet'}")


def extract_data(
    pipeline_path: str,
    extract_periods: list[str],
    config: dict,
    dhis2_snis_client: DHIS2,
    run_task: bool,
) -> None:
    """Retrieves DHIS2 analytics data elements and reporting rates for the given periods.

    Args:
        pipeline_path (str): Root path of the pipeline used to resolve input/output data folders.
        extract_periods (list[str]): Periods to extract, in YYYYMM format.
        config (dict): Extraction configuration loaded from extract_config.json.
        dhis2_snis_client (DHIS2): Connected DHIS2 client used to retrieve the data.
        run_task (bool): Whether to run this extraction step.
    """
    if not run_task:
        current_run.log_info("Skipping data extraction as run_task is set to False.")
        return

    current_run.log_info("Retrieving DHIS2 analytics data")

    # retrieve FOSA ids from SNIS
    fosa_list = _get_ou_list(
        pyramid_fname=pipeline_path / "data" / "pyramid_metadata" / "snis_pyramid_metadata.parquet", ou_level=5
    )
    current_run.log_info(f"Download MODE: {config['SETTINGS']['MODE']} for periods: {extract_periods}")

    # limits
    dhis2_snis_client.analytics.MAX_DX = 100
    dhis2_snis_client.analytics.MAX_ORG_UNITS = 100
    dhis2_snis_client.data_value_sets.MAX_DATA_ELEMENTS = 100
    dhis2_snis_client.data_value_sets.MAX_ORG_UNITS = 100

    _extract_data_elements_for_periods(
        pipeline_path=pipeline_path,
        dhis2_client=dhis2_snis_client,
        periods=extract_periods,
        org_unit_list=fosa_list,
        config=config,
    )
    current_run.log_info("Data elements extract finished.")

    _extract_reporting_rates_for_periods(
        pipeline_path=pipeline_path,
        dhis2_client=dhis2_snis_client,
        periods=extract_periods,
        org_unit_list=fosa_list,
        config=config,
    )
    current_run.log_info("Reporting rates extract finished.")


def _get_ou_list(pyramid_fname: Path, ou_level: int) -> list:
    """Retrieves a list of organizational unit IDs from the pyramid Parquet file based on the specified OU level.

    Args:
        pyramid_fname (Path): Path to the pyramid metadata Parquet file.
        ou_level (int): Organisation unit level to filter by.

    Returns:
        list: Organisation unit IDs corresponding to the specified OU level.
    """
    try:
        # Retrieve organisational units and filter by ou_level
        ous = pl.read_parquet(pyramid_fname)
        ou_list = ous.filter(pl.col("level") == ou_level)["id"].to_list()
    except Exception as e:
        raise Exception(f"Error loading pyramid file: {e}") from e

    current_run.log_info(f"DHIS2 org units id list {len(ou_list)} at level {ou_level}")
    return ou_list


def _extract_data_elements_for_periods(
    pipeline_path: Path,
    dhis2_client: DHIS2,
    periods: list[str],
    org_unit_list: list[str],
    config: dict,
) -> None:
    """Downloads data elements for each period, with error handling and logging.

    Args:
        pipeline_path (Path): Root path of the pipeline used to resolve the output data folder.
        dhis2_client (DHIS2): Connected DHIS2 client used to retrieve the data.
        periods (list[str]): Periods to extract, in YYYYMM format.
        org_unit_list (list[str]): Organisation unit IDs to extract data for.
        config (dict): Extraction configuration loaded from extract_config.json.
    """
    # Setup extractor
    dhis2_extractor = DHIS2Extractor(dhis2_client=dhis2_client, download_mode=config["SETTINGS"]["MODE"])
    try:
        for period in periods:
            raw_data_path = dhis2_extractor.data_elements.download_period(
                data_elements=config["DATA_ELEMENTS"]["UIDS"],
                org_units=org_unit_list,
                period=period,
                output_dir=pipeline_path / "data" / "data_elements",
                filename=f"data_{period}.parquet",
            )
            if not raw_data_path:
                current_run.log_info(f"No data elements data for period {period}.")
    except Exception as e:
        raise Exception(f"Extract data elements error : {e}") from e  # let it crash!


def _extract_reporting_rates_for_periods(
    pipeline_path: Path,
    dhis2_client: DHIS2,
    periods: list[str],
    org_unit_list: list[str],
    config: dict,
) -> None:
    """Downloads reporting rates for each period, with error handling and logging.

    Args:
        pipeline_path (Path): Root path of the pipeline used to resolve the output data folder.
        dhis2_client (DHIS2): Connected DHIS2 client used to retrieve the data.
        periods (list[str]): Periods to extract, in YYYYMM format.
        org_unit_list (list[str]): Organisation unit IDs to extract data for.
        config (dict): Extraction configuration loaded from extract_config.json.
    """
    # Setup extractor
    dhis2_extractor = DHIS2Extractor(dhis2_client=dhis2_client, download_mode=config["SETTINGS"]["MODE"])
    rr_ids = config["REPORTING_RATES"].get("DATASETS", [])
    rr_metrics = config["REPORTING_RATES"].get("METRICS", {}).keys()
    reporting_combinations = [f"{ds}.{metric}" for ds, metric in product(rr_ids, rr_metrics)]

    try:
        for period in periods:
            raw_data_path = dhis2_extractor.reporting_rates.download_period(
                reporting_rates=reporting_combinations,
                org_units=org_unit_list,
                period=period,
                output_dir=pipeline_path / "data" / "reporting_rates",
                filename=f"data_{period}.parquet",
            )
            if not raw_data_path:
                current_run.log_info(f"No reporting rates data for period {period}.")
    except Exception as e:
        raise Exception(f"Extract reporting rates error : {e}") from e  # let it crash!


def compile_palu_extracts(
    extract_periods: list[str],
    data_path: Path,
    snis_extracts_path: Path,
    output_path: Path,
    config_path: Path,
    run_task: bool,
) -> list[Path]:
    """Collects and creates extracts based on the new extracts and searches for required data in snis extracts.

    Args:
        extract_periods (list[str]): Periods to compile, in YYYYMM format.
        data_path (Path): Path to this pipeline's own extracted data.
        snis_extracts_path (Path): Path to the dhis2_snis_extract pipeline's data.
        output_path (Path): Path where the compiled palu extracts are saved.
        config_path (Path): Path to the folder containing required_snis_ids.py.
        run_task (bool): Whether to run this compilation step.

    Returns:
        list[Path]: Paths of the compiled palu extracts, including the pyramid metadata and population data.
    """
    if not run_task:
        current_run.log_info("Skipping palu extracts compilation as run_task is set to False.")
        return []

    current_run.log_info("Compiling palu extracts..")
    output_path.mkdir(parents=True, exist_ok=True)

    palu_extracts = []
    palu_extracts.append(data_path / "pyramid_metadata" / "snis_pyramid_metadata.parquet")
    req_de, req_rr, req_rr_metrics = load_required_dhis2_uids(config_path / "required_snis_ids.py")

    extract_path = collect_data_for_periods(
        periods=extract_periods,
        source_path=data_path,
        snis_extracts_path=snis_extracts_path,
        output_path=output_path,
        required_data_elements=req_de,
        required_reporting_rates=req_rr,
        required_reporting_metrics=req_rr_metrics,
    )
    palu_extracts.extend(extract_path)

    pop_paths = collect_population_data_for_periods(
        extract_periods=extract_periods,
        snis_extracts_path=snis_extracts_path,
    )
    palu_extracts.extend(pop_paths)

    return palu_extracts


def load_required_dhis2_uids(identifiers_fname: Path) -> tuple[list[str], list[str], list[str]]:
    """Loads the required DHIS2 data identifiers from a Python config file.

    Args:
        identifiers_fname (Path): Path to the Python file defining the required UID lists.

    Returns:
        tuple[list[str], list[str], list[str]]: Required data elements, required reporting rates,
            and required reporting metrics, in that order.
    """
    namespace = {}
    exec(identifiers_fname.read_text(encoding="utf-8"), namespace)
    return (
        namespace["required_data_elements"],
        namespace["required_reporting_rates"],
        namespace["required_reporting_metrics"],
    )


def collect_data_for_periods(
    periods: list[str],
    source_path: Path,
    snis_extracts_path: Path,
    output_path: Path,
    required_data_elements: list,
    required_reporting_rates: list,
    required_reporting_metrics: list,
) -> list[Path]:
    """Collects and creates extracts based on the new extracts and searches for additional data in snis extracts.

    Args:
        periods (list[str]): Periods to compile, in YYYYMM format.
        source_path (Path): Path to this pipeline's own extracted data.
        snis_extracts_path (Path): Path to the dhis2_snis_extract pipeline's data.
        output_path (Path): Path where the compiled palu extracts are saved.
        required_data_elements (list): Data element UIDs to include from the SNIS extracts.
        required_reporting_rates (list): Reporting rate UIDs to include from the SNIS extracts.
        required_reporting_metrics (list): Reporting rate metrics to include from the SNIS extracts.

    Returns:
        list[Path]: Paths of the compiled palu extracts.
    """
    current_run.log_info(f"Compiling palu extract for period: {periods}..")

    # Set up the schema for the extract DataFrame
    extract_schema = {
        "data_type": pl.String,
        "dx": pl.String,
        "period": pl.String,
        "org_unit": pl.String,
        "category_option_combo": pl.String,
        "attribute_option_combo": pl.String,
        "rate_metric": pl.String,
        "domain_type": pl.String,
        "value": pl.String,
    }

    palu_extracts = []
    for period in periods:
        data_elements_snis_file = next((snis_extracts_path / "snis_extracts").glob(f"snis_data_{period}.parquet"), None)
        snis_df = pl.read_parquet(data_elements_snis_file) if data_elements_snis_file else pl.DataFrame()

        data_elements_df = _collect_data_elements_for_period(
            period=period,
            source_path=source_path / "data_elements",
            snis_extract=snis_df,
            snis_required_de=required_data_elements,
            schema=extract_schema,
        )
        reporting_rates_df = _collect_reporting_rates_for_period(
            period=period,
            source_path=source_path / "reporting_rates",
            snis_extract=snis_df,
            snis_required_rr=required_reporting_rates,
            snis_required_metrics=required_reporting_metrics,
            schema=extract_schema,
        )

        palu_extract_df = pl.concat([data_elements_df, reporting_rates_df])
        if palu_extract_df.is_empty():
            current_run.log_info(f"No data found for period {period}. Skipping extract.")
            continue

        save_to_parquet(data=palu_extract_df, filename=output_path / f"palu_extract_{period}.parquet")
        current_run.log_info(
            f"Palu extract for period {period} saved at {output_path / f'palu_extract_{period}.parquet'}"
        )
        palu_extracts.append(output_path / f"palu_extract_{period}.parquet")

    return palu_extracts


def _collect_data_elements_for_period(
    period: str, source_path: Path, snis_extract: pl.DataFrame, snis_required_de: list, schema: dict
) -> pl.DataFrame:
    """Collects data elements for a given period from the source path and appends them to the provided DataFrame.

    Also searches for additional data elements in the SNIS extracts and appends them to the DataFrame.

    Args:
        period (str): Period to collect, in YYYYMM format.
        source_path (Path): Path to the local pipeline data folder for data elements.
        snis_extract (pl.DataFrame): SNIS extract data to search for additional data elements.
        snis_required_de (list): Data element UIDs to include from the SNIS extract.
        schema (dict): Polars schema used to cast the collected data.

    Returns:
        pl.DataFrame: Collected data elements for the specified period.
    """
    # Search in local pipeline data folder
    data_elements_df = pl.DataFrame(schema=schema)
    data_elements_file = next(source_path.glob(f"data_{period}.parquet"), None)
    if data_elements_file:
        data_elements_df = pl.read_parquet(data_elements_file).cast(schema)

    # Search for the additional data elements in parquet files in the snis folder
    if not snis_extract.is_empty():
        snis_de_df = snis_extract.filter(
            (pl.col("data_type") == "DATA_ELEMENT") & pl.col("dx").is_in(snis_required_de)
        ).cast(schema)
        data_elements_df = pl.concat([data_elements_df, snis_de_df])

    return data_elements_df


def _collect_reporting_rates_for_period(
    period: str,
    source_path: Path,
    snis_extract: pl.DataFrame,
    snis_required_rr: list,
    snis_required_metrics: list,
    schema: dict,
) -> pl.DataFrame:
    """Collects reporting rates for a given period from the source path and appends them to the provided DataFrame.

    Also searches for additional reporting rates in the SNIS extracts and appends them to the DataFrame.

    Args:
        period (str): Period to collect, in YYYYMM format.
        source_path (Path): Path to the local pipeline data folder for reporting rates.
        snis_extract (pl.DataFrame): SNIS extract data to search for additional reporting rates.
        snis_required_rr (list): Reporting rate UIDs to include from the SNIS extract.
        snis_required_metrics (list): Reporting rate metrics to include from the SNIS extract.
        schema (dict): Polars schema used to cast the collected data.

    Returns:
        pl.DataFrame: Collected reporting rates for the specified period.
    """
    # Search in local pipeline data folder
    reporting_rates_df = pl.DataFrame(schema=schema)
    reporting_rates_file = next(source_path.glob(f"data_{period}.parquet"), None)
    if reporting_rates_file:
        reporting_rates_df = pl.read_parquet(reporting_rates_file).cast(schema)

    # Search for the additional reporting rates in parquet files in the snis folder
    if not snis_extract.is_empty():
        snis_rr_df = snis_extract.filter(
            (pl.col("data_type") == "REPORTING_RATE")
            & pl.col("dx").is_in(snis_required_rr)
            & (pl.col("rate_metric").is_in(snis_required_metrics))
        ).cast(schema)
        reporting_rates_df = pl.concat([reporting_rates_df, snis_rr_df])

    return reporting_rates_df


def collect_population_data_for_periods(extract_periods: list[str], snis_extracts_path: Path) -> list[Path]:
    """Collects population data for the specified periods from the SNIS extracts.

    Args:
        extract_periods (list[str]): Periods to collect, in YYYYMM format.
        snis_extracts_path (Path): Path to the dhis2_snis_extract pipeline's data.

    Returns:
        list[Path]: Paths of the population data extracts found for each period.
    """
    pop_paths = []
    year_periods = sorted(set([p[0:4] for p in extract_periods]))
    for period in year_periods:
        pop_file = next((snis_extracts_path / "population").glob(f"snis_population_{period}.parquet"), None)
        if pop_file:
            pop_paths.append(pop_file)
        else:
            current_run.log_info(f"No population data found for period {period}.")
    return pop_paths


def update_snis_dataset(
    pipeline_path: Path,
    snis_extracts_path: Path,
    new_extracts: list[Path],
    extract_periods: list,
    dataset_id: str,
    run_task: bool,
) -> None:
    """Updates the SNIS dataset with the new extracts.

    Args:
        pipeline_path (Path): Root path of the pipeline, used to locate its data directory.
        snis_extracts_path (Path): Path to the dhis2_snis_extract pipeline's data.
        new_extracts (list[Path]): Paths of the new extract files to push to the dataset.
        extract_periods (list): Periods (YYYYMM) to look up population and data extracts for.
        dataset_id (str): OpenHEXA dataset identifier to update.
        run_task (bool): Whether to run this update step.
    """
    if not run_task:
        return

    if not new_extracts:
        current_run.log_info("No new extracts, loading data from repository folder.")
        new_extracts = build_snis_extracts_list(pipeline_path, snis_extracts_path, extract_periods)

    try:
        add_files_to_dataset(
            dataset_id=dataset_id,
            file_paths=new_extracts,
            ds_version_prefix="SNIS_palu_mensuel",
        )
    except Exception as e:
        raise Exception(f"Error while updating SNIS dataset: {e}") from e


def resolve_extract_periods(start_date: str, end_date: str, config: dict) -> list:
    """Resolves the extract periods based on the provided start and end dates.

    It also validates them against the configuration.

    Returns
    -------
    list
        A list of extract periods in YYYYMM format.
    """
    try:
        start, end = resolve_dates_and_validate(start_date, end_date, config)
        return get_extract_periods(start, end)
    except Exception as e:
        current_run.log_error(f"Error resolving extract periods: {e}")
        raise


def build_snis_extracts_list(pipeline_path: Path, snis_extracts_path: Path, extract_periods: list) -> list:
    """Builds a list of SNIS extract file paths found in the pipeline's data directory.

    Args:
        pipeline_path: Root path of the pipeline, used to locate its data directory.
        snis_extracts_path: Path to the dhis2_snis_extract pipeline's data.
        extract_periods: Periods (YYYYMM) to look up population and data extracts for.

    Returns:
        list: File paths for the pyramid, population, and data extracts that exist on disk.
    """
    extracts_list = []
    snis_pyramid_dir = pipeline_path / "data" / "pyramid_metadata" / "snis_pyramid_metadata.parquet"
    if snis_pyramid_dir.exists():
        extracts_list.append(snis_pyramid_dir)

    # Palu extract files
    snis_palu_extracts_dir = pipeline_path / "data" / "palu_extracts"
    if snis_palu_extracts_dir.exists():
        for period in extract_periods:
            snis_palu_extract_file = snis_palu_extracts_dir / f"palu_extract_{period}.parquet"
            if snis_palu_extract_file.exists():
                extracts_list.append(snis_palu_extract_file)

    # NOTE: Load populaton from dhis2_snis_extract pipeline data folder
    snis_population_dir = snis_extracts_path / "population"
    if snis_population_dir.exists():
        for year in sorted(set([p[0:4] for p in extract_periods])):
            snis_population_file = snis_population_dir / f"snis_population_{year}.parquet"
            if snis_population_file.exists():
                extracts_list.append(snis_population_file)

    return extracts_list


if __name__ == "__main__":
    dhis2_snis_palu_data_mensuel()
