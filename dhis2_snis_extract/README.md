# DHIS2 SNIS Extract

This pipeline is dedicated to extracting routine health data from the DRC SNIS DHIS2 instance. The extracted data is shared to the PNLP workspace on OpenHEXA, where it is then integrated into the PNLP DHIS2.

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `start_date` | string | Auto | Start of the extraction window in `YYYYMM` format. Defaults to current date minus a configured number of months. |
| `end_date` | string | Auto | End of the extraction window in `YYYYMM` format. Defaults to current date minus 1 month. |
| `run_orgunits` | bool | `true` | Extract the organisation unit pyramid from SNIS. |
| `run_pop` | bool | `true` | Extract population data per year. |
| `run_analytics` | bool | `true` | Extract routine data, reporting rates, and ACM indicators. |
| `add_to_dataset` | bool | `true` | Push all extracts produced in this run to the OpenHEXA dataset. |

## Output dataset

Extracts are added as a new version to the **`snis-extracts`** dataset (version prefix: `SNIS_extract`).

## Output files

All files are saved in **Parquet** format.

- **Org unit pyramid** — one file covering all organisation units up to level 5, including hierarchy, geometry, and opening/closing dates.
- **Population** — one file per year (e.g. `snis_population_2024.parquet`), containing population values at *aire de santé* level (OU level 4).
- **Monthly SNIS extract** — one file per month (e.g. `snis_data_202503.parquet`), merging routine data elements, reporting rates, and ACM indicators at FOSA level (OU level 5). This is the main output consumed downstream.
