import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import polars as pl
from d2d_development.extract import DHIS2Extractor
from dateutil.relativedelta import relativedelta
from openhexa.sdk import current_run, parameter, pipeline, workspace
from openhexa.toolbox.dhis2 import DHIS2
from openhexa.toolbox.dhis2.dataframe import get_organisation_unit_groups
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
    code="run_metadata_data",
    name="Extract metadata data",
    type=bool,
    default=True,
    help="Extract metadata from source DHIS2.",
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
    help="If True, skips downloading extracts that already exist on disk.",
    default=False,
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
    start_date: str,
    end_date: str,
    run_metadata_data: bool,
    run_extract_data: bool,
    skip_existing_extracts: bool,
    add_to_dataset: bool,
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
        run_metadata_data: If True, runs the pyramid and organisation unit group extraction tasks.
        run_extract_data: If True, runs the pyramid and data extraction tasks.
        skip_existing_extracts: If True, skips downloading extracts that already exist on disk.
        add_to_dataset: If True, adds the extracted data to the dataset.

    Raises:
        Exception: If an error occurs during the pipeline execution.
    """
    pipeline_path = Path(workspace.files_path) / "pipelines" / "dhis2_snis_prs_cmm_morbidity_extract"
    config = read_json_file(pipeline_path / "configuration" / "extract_config.json")
    source_dhis2 = connect_to_dhis2(connection_str=config["SETTINGS"]["SOURCE_DHIS2_CONNECTION"])
    updates_collector = {}

    try:
        extract_metadata(
            pipeline_path=pipeline_path,
            source_dhis2=source_dhis2,
            config=config,
            sync_config=read_json_file(pipeline_path / "configuration" / "sync_config.json"),
            updates_collector=updates_collector,
            run_task=run_metadata_data,
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
            start_date=start_date,
            end_date=end_date,
            config=config,
            updates_collector=updates_collector,
            dataset_id="snis-prs-cmm-extract",
            run_task=add_to_dataset,
        )
    except Exception as e:
        current_run.log_error(f"An error occurred while updating the dataset: {e}")
        raise


def extract_metadata(
    pipeline_path: Path, source_dhis2: DHIS2, config: dict, sync_config: dict, updates_collector: dict, run_task: bool
) -> None:
    """Extracts the source DHIS2 pyramid and organisation unit groups and saves them as Parquet files."""
    extract_pyramid(
        dhis2_client=source_dhis2,
        sync_config=sync_config,
        output_dir=pipeline_path / "data" / "pyramid",
        filename="pyramid_data.parquet",
        updates_collector=updates_collector,
        run_task=run_task,
    )

    extract_org_unit_groups(
        dhis2_client=source_dhis2,
        config=config,
        output_dir=pipeline_path / "data" / "org_unit_groups",
        updates_collector=updates_collector,
        run_task=run_task,
    )


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
    updates_collector.setdefault("pyramid", []).append(pyramid_fname)
    current_run.log_info(f"DHIS2 pyramid data saved: {pyramid_fname}")


def extract_org_unit_groups(
    dhis2_client: DHIS2,
    config: dict,
    output_dir: Path,
    updates_collector: dict,
    run_task: bool,
):
    """Extracts the source DHIS2 organisation unit groups and saves them as a Parquet file.

    Args:
        dhis2_client: DHIS2 client instance for data extraction.
        config: Configuration dictionary containing extraction settings.
        output_dir: Directory where the org unit groups Parquet file will be saved.
        updates_collector: Dictionary to collect paths of new extracts produced in this run.
        run_task: If False, the org unit group extraction is skipped entirely.
    """
    if not run_task:
        current_run.log_info("Organisation unit group extraction skipped.")
        return

    source_oug_id = config.get("ORG_UNIT_GROUPS", {}).get("OUG_URBAN", "")
    if not source_oug_id:
        current_run.log_warning("No org unit group configured in extract_config.json; skipping.")
        return

    oug_source = get_organisation_unit_groups(dhis2_client)
    source_oug = oug_source.filter(pl.col("id").is_in([source_oug_id]))
    if source_oug.is_empty():
        current_run.log_warning(f"Org unit group '{source_oug_id}' not found in source DHIS2; nothing saved.")
        return

    save_to_parquet(data=source_oug, filename=output_dir / "org_unit_groups.parquet")
    updates_collector.setdefault("org_unit_groups", []).append(output_dir / "org_unit_groups.parquet")
    current_run.log_info(f"Organisation unit groups ({source_oug_id}) saved: {output_dir / 'org_unit_groups.parquet'}")


def extract_prs_data(
    pipeline_path: Path,
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

    # Load the source datasets and pyramid data (Already filtered)
    source_pyramid = pl.read_parquet(pipeline_path / "data" / "pyramid" / "pyramid_data.parquet")

    # get dates and validate
    start, end = resolve_dates_and_validate(start_date, end_date, config)

    # NOTE: Adjust the start date to consider the 6 months CMM windows
    cmm_window = config["SETTINGS"].get("CMM_MONTHS_WINDOW", 6)
    start_cmm = (datetime.strptime(start, "%Y%m") - relativedelta(months=cmm_window)).strftime("%Y%m")
    extract_periods = get_extract_periods(start_cmm, end)

    # Set extractor limits
    # dhis2_client.data_value_sets.MAX_DATA_ELEMENTS = 100
    # dhis2_client.data_value_sets.MAX_ORG_UNITS = 100
    current_run.log_info(f"Extract periods from: {start_cmm} ({cmm_window} cmm window) to {end}")
    handle_data_element_extracts(
        pipeline_path=pipeline_path,
        dhis2_client=dhis2_client,
        config=config,
        extract_periods=extract_periods,
        source_pyramid=source_pyramid,
        updates_collector=updates_collector,
        skip_existing_extracts=skip_existing_extracts,
    )

    current_run.log_info("Extracts finished.")


def get_fosa_descendants_of_zs(pyramid: pl.DataFrame, dhis2_client: DHIS2, oug_id: str) -> list:
    """Retrieves the list of FOSA organisation units that are descendants of urban Zones de sante.

    Parameters
    ----------
    pyramid : pl.DataFrame
        The organisation units pyramid as a Polars DataFrame.
    dhis2_client : DHIS2
        The DHIS2 client instance.
    oug_id : str
        The organisation unit group ID for urban Zones de sante.

    Returns
    -------
    list
        List of level 5 organisation unit IDs that are descendants of urban Zones de sante.
    """
    current_run.log_info(f"Retrieving Organization Units for Urban Health Zones under OUG '{oug_id}'")
    ou_groups = get_organisation_unit_groups(dhis2_client)
    zs_urban = ou_groups.filter(pl.col("id") == oug_id)
    zs_urban_list = zs_urban["organisation_units"].explode().to_list()
    parent_map = dict(
        zip(
            pyramid["id"],
            pyramid["parent"].apply(lambda x: x["id"] if isinstance(x, dict) else None),
            strict=True,
        )
    )
    level5 = pyramid[pyramid["level"] == 5]["id"]

    def get_zs_parent(ou: str) -> str | None:
        """Climb 5 → 4 → 3.

        Returns:
          level 3 parent of level 5 org unit.
        """
        p4 = parent_map.get(ou)
        if not p4:
            return None
        return parent_map.get(p4)

    return [ou for ou in level5 if get_zs_parent(ou) in zs_urban_list]


def handle_data_element_extracts(
    pipeline_path: Path,
    dhis2_client: DHIS2,
    config: dict,
    extract_periods: list[str],
    source_pyramid: pl.DataFrame,
    updates_collector: dict,
    skip_existing_extracts: bool,
):
    """Handles data elements extracts based on the configuration."""
    data_element_extracts = config.get("DATA_ELEMENTS", {}).get("EXTRACTS", {})
    if len(data_element_extracts) == 0:
        current_run.log_info("No data elements to extract.")
        return

    current_run.log_info("Starting data element extracts.")

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

        # get org units from the filtered pyramid
        org_units = source_pyramid.filter(pl.col("level") == org_units_level).get_column("id").to_list()
        current_run.log_info(
            f"Starting data elements extract ID: '{extract_id}' ({idx + 1}) "
            f"with {len(data_element_uids)} data elements across {len(org_units)} org units "
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
    pipeline_path: Path,
    start_date: str,
    end_date: str,
    config: dict,
    updates_collector: dict[Path],
    dataset_id: str,
    run_task: bool,
) -> None:
    """Updates the SNIS dataset with the new extracts.

    This function takes the paths of the new extracts from the updates collector and updates the OH dataset.
    NOTE: Additionally includes a json file to link the extract files with their extract_id required for integration.

    Args:
        pipeline_path: Root path of the pipeline, used to save the extract/id mapping file.
        start_date: Start date for data extraction in YYYYMM format.
        end_date: End date for data extraction in YYYYMM format.
        config: Configuration dictionary containing extraction settings.
        updates_collector: Dictionary of file paths produced in this run, keyed by extract identifier.
        dataset_id: The ID of the OpenHEXA dataset to update.
        run_task: If False, the dataset update is skipped entirely.
    """
    if not run_task:
        current_run.log_info("Dataset update skipped.")
        return

    new_extracts = [item for values in updates_collector.values() for item in values]
    mapping_file_path = pipeline_path / "data" / "updates_collector.json"

    if not new_extracts:
        current_run.log_info("No new extracts, loading data from repository folder.")
        # Replicate the logic to get the extract periods considering cmm window
        cmm_window = config["SETTINGS"].get("CMM_MONTHS_WINDOW", 6)
        start, end = resolve_dates_and_validate(start_date, end_date, config)
        start_cmm = (datetime.strptime(start, "%Y%m") - relativedelta(months=cmm_window)).strftime("%Y%m")
        extract_periods = get_extract_periods(start_cmm, end)
        new_extracts, updates_collector = build_extracts_list(
            pipeline_path,
            extract_periods,
            extract_id="fosa_morbidity",  # hardcoded for now, as is the only one
        )

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


def build_extracts_list(pipeline_path: Path, extract_periods: list, extract_id: str) -> tuple[list[Path], dict]:
    """Builds a list of SNIS extract file paths found in the pipeline's data directory.

    NOTE: So we can add files to the dataset even if no new extracts were produced in this run.

    Args:
        pipeline_path: Root path of the pipeline, used to locate its data directory.
        extract_periods: Periods (YYYYMM) to look up population and data extracts for.
        extract_id: Identifier for the data extract, used to locate its directory.

    Returns:
        tuple[list[Path], dict]: A tuple containing:
            - A list of Path objects for the found extract files.
            - A dictionary mapping extract identifiers to lists of their corresponding file paths.
    """
    snis_pyramid_path = pipeline_path / "data" / "pyramid" / "pyramid_data.parquet"
    snis_oug_path = pipeline_path / "data" / "org_unit_groups" / "org_unit_groups.parquet"
    snis_extracts_path = pipeline_path / "data" / "extracts" / "data_elements" / extract_id
    extracts_list = []
    updates_file = {}

    if snis_pyramid_path.exists():
        extracts_list.append(snis_pyramid_path)
        updates_file.setdefault("pyramid", []).append(snis_pyramid_path)
    else:
        current_run.log_warning(f"Pyramid file not found: {snis_pyramid_path}")

    if snis_oug_path.exists():
        extracts_list.append(snis_oug_path)
        updates_file.setdefault("org_unit_groups", []).append(snis_oug_path)
    else:
        current_run.log_warning(f"Organisation unit groups file not found: {snis_oug_path}")

    if snis_extracts_path.exists():
        for period in extract_periods:
            snis_extract_file = snis_extracts_path / f"data_{period}.parquet"
            if snis_extract_file.exists():
                extracts_list.append(snis_extract_file)
                updates_file.setdefault(extract_id, []).append(snis_extract_file)
            else:
                current_run.log_warning(f"Extract file for period {period} not found: {snis_extract_file}")

    return extracts_list, updates_file


if __name__ == "__main__":
    dhis2_snis_prs_cmm_morbidity_extract()
