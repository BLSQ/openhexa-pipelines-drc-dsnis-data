# DHIS2 SNIS PRS CMM Morbidity Extract

This pipeline extracts morbidity data elements from the DRC SNIS DHIS2 instance for all FOSA (facility-level, OU level 5) under a configured list of provinces, used to compute the CMM (Consommation Moyenne Mensuelle) for the PRS integration. The extraction window is automatically extended backward by a configurable CMM window so downstream processes can compute the rolling average. The extracted data is shared to the DRC PRS Workspace for integration via dataset.

> **Note:** Org units and data elements to extract are driven by the `sync_config.json` and `extract_config.json` configuration files.

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `start_date` | string | Auto | Start of the extraction window in `YYYYMM` format. Defaults to current date minus a configured number of months. |
| `end_date` | string | Auto | End of the extraction window in `YYYYMM` format. Defaults to current date minus 1 month. |
| `run_metadata_data` | bool | `true` | Extract the org unit pyramid and org unit groups from the source DHIS2. |
| `run_extract_data` | bool | `true` | Extract morbidity data elements from the source DHIS2. |
| `skip_existing_extracts` | bool | `false` | If `true`, skips downloading extracts that already exist on disk. |
| `add_to_dataset` | bool | `true` | Push all extracts produced in this run to the OpenHEXA dataset. |

## Output dataset

Extracts are added as a new version to the **`snis-prs-cmm-extract`** dataset (version prefix: `PRS_DS_CMM_extract`).

## Output files

All files are saved in **Parquet** format.

- **Org unit pyramid** (`pyramid_data.parquet`) — organisation units up to level 5, restricted to the selected provinces (`sync_config.json`) and all their descendants.
- **Org unit groups** (`org_unit_groups.parquet`) — the urban Zones de Santé org unit group (`extract_config.json`).
- **Data element extracts** — one file per month (e.g. `data_{YYYYMM}.parquet`) per extract configuration (currently `fosa_morbidity`), covering the requested window extended backward by the configured CMM window (default 6 months).
- **Mapping file** (`updates_collector.json`) — a JSON file included in every dataset version that maps each extract file name to its extract identifier, used by downstream integration processes.
