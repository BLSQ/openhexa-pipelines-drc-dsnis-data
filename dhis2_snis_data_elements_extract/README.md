# DHIS2 SNIS Data Elements Extract

This pipeline is dedicated to extracting raw data elements from the DRC SNIS DHIS2 instance. The extracted data is shared to the PNLP workspace on OpenHEXA, where it is then integrated into the PNLP DHIS2.

> **Note:** This pipeline only supports periods from January 2025 onward (`202501` minimum).

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `start_date` | string | Auto | Start of the extraction window in `YYYYMM` format (minimum `202501`). Defaults to current date minus a configured number of months. |
| `end_date` | string | Auto | End of the extraction window in `YYYYMM` format (minimum `202501`). Defaults to current date minus 1 month. |
| `run_orgunits` | bool | `true` | Extract the organisation unit pyramid from SNIS. |
| `run_extract_data` | bool | `true` | Extract data elements from the source DHIS2. |
| `add_to_dataset` | bool | `true` | Push all extracts produced in this run to the OpenHEXA dataset. |

## Output dataset

Extracts are added as a new version to the **`snis-data-elements-extracts`** dataset (version prefix: `SNIS_DE_extract`).

## Output files

All files are saved in **Parquet** format.

- **Org unit pyramid** — one file covering all organisation units up to level 5, including hierarchy, geometry, and opening/closing dates.
- **Monthly data element extract** — one file per month (e.g. `data_202503.parquet`), containing raw data element values at FOSA level (OU level 5) for the data elements configured in the pipeline.
