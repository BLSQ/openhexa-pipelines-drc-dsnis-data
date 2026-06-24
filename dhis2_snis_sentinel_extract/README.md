# DHIS2 SNIS Sentinel Extract

This pipeline is dedicated to extracting sentinel site data from the DRC SNIS DHIS2 instance. The extracted data is shared to the PNLP workspace on OpenHEXA, where it is then integrated into the PNLP DHIS2.

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `start_date` | string | Auto | Start of the extraction window in `YYYYMM` format. Defaults to current date minus a configured number of months. |
| `end_date` | string | Auto | End of the extraction window in `YYYYMM` format. Defaults to current date minus 1 month. |
| `run_extract_data` | bool | `true` | Extract data elements from the source DHIS2. |
| `add_to_dataset` | bool | `true` | Push all extracts produced in this run to the OpenHEXA dataset. |

## Output dataset

Extracts are added as a new version to the **`snis-sentinel-dataset`** dataset (version prefix: `SNIS_SENTINEL_extract`).

## Output files

All files are saved in **Parquet** format.

- **Data element extracts** — one file per extract configuration per month (e.g. `data_{extract_id}_{YYYYMM}.parquet`). Each extract corresponds to a specific sentinel DHIS2 dataset defined in the pipeline configuration, covering the org units associated with that dataset.
- **Mapping file** (`updates_collector.json`) — a JSON file included in every dataset version that maps each extract file name to its extract identifier, used by downstream integration processes.
