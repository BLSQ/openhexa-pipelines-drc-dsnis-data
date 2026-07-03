# DHIS2 SNIS Palu Data Mensuel

This pipeline is dedicated to compiling monthly malaria (palu) data from the DRC SNIS DHIS2 instance. The extracted data is shared to the RDC Palu rapports mensuels (rdc-palu-rapports-mensuels) workspace on OpenHEXA, where it is then used for reporting.

The pipeline combines two sources: data elements and reporting rates extracted directly from SNIS, supplemented by data pulled from the outputs of the **DHIS2 SNIS Extract** pipeline. Only periods from January 2025 onward are supported.

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `start_date` | string | Auto | Start of the extraction window in `YYYYMM` format (minimum `202501`). Defaults to current date minus a configured number of months. |
| `end_date` | string | Auto | End of the extraction window in `YYYYMM` format (minimum `202501`). Defaults to current date minus 1 month. |
| `run_extract_data` | bool | `true` | Extract pyramid metadata, data elements and reporting rates from SNIS. |
| `add_to_dataset` | bool | `true` | Push all extracts produced in this run to the OpenHEXA dataset. |

## Output dataset

Extracts are added as a new version to the **`snis-palu-mensuel-extracts`** dataset (version prefix: `SNIS_palu_mensuel`).

## Output files

All files are saved in **Parquet** format.

- **Org unit pyramid** — one file with FOSA-level (OU level 5) metadata used for alignment downstream.
- **Monthly palu extract** — one file per month (e.g. `palu_extract_202503.parquet`), merging palu-specific data elements and reporting rates from both the direct SNIS extraction and the SNIS Extract pipeline outputs.
- **Population** — yearly population files sourced from the SNIS Extract pipeline outputs, included as-is for downstream use.
