# DHIS2 PRS Dataset Extract

This pipeline extracts a subset of the DRC SNIS DHIS2 data scoped for the PRS integration. It extracts data elements at facility (level 5) and zone de santé (level 3) levels, restricted to organisation units under a configured list of provinces, along with their org unit pyramid. The extracted data is shared to the PRS DHIS2 instance for integration.

> **Note:** Org units and data elements to extract are driven by the `sync_config.json` and `extract_config.json` configuration files.

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `start_date` | string | Auto | Start of the extraction window in `YYYYMM` format. Defaults to current date minus a configured number of months. |
| `end_date` | string | Auto | End of the extraction window in `YYYYMM` format. Defaults to current date minus 1 month. |
| `run_extract_data` | bool | `true` | Extract the org unit pyramid and data elements from the source DHIS2. |
| `add_to_dataset` | bool | `true` | Push all extracts produced in this run to the OpenHEXA dataset. |

## Output dataset

Extracts are added as a new version to the **`snis-prs-dataset-sync`** dataset (version prefix: `PRS_DS_SYNC_extract`).

## Output files

All files are saved in **Parquet** format.

- **Org unit pyramid** (`snis_prs_pyramid.parquet`) — organisation units up to the configured limit level, restricted to the selected provinces (`sync_config.json`) and all their descendants.
- **Data element extracts** — one file per extract configuration per month (e.g. `data_{extract_id}_{YYYYMM}.parquet`). Each extract is scoped to a specific organisation unit level (facility or zone de santé) as defined in `extract_config.json`.
- **Mapping file** (`updates_collector.json`) — a JSON file included in every dataset version that maps each extract file name to its extract identifier, used by downstream integration processes.
