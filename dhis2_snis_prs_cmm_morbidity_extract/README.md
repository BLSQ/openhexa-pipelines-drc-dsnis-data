# DHIS2 SNIS PRS CMM Morbidity Extract

This pipeline extracts morbidity data elements from the DRC SNIS DHIS2 instance for all FOSA (facility-level, OU level 5) under a configured list of provinces, used to compute the CMM (Consommation Moyenne Mensuelle) for the PRS integration. The extraction window is automatically extended backward by a configurable CMM window so downstream processes can compute the rolling average. The extracted data is shared to the DRC PRS Workspace for integration via dataset.

> **Note:** Org units and data elements to extract are driven by the `sync_config.json` and `extract_config.json` configuration files.

## Configuration files

`sync_config.json` lists the province org unit UIDs to sync the pyramid for, and the org unit group(s) to extract:

```json
{
    "ORG_UNITS": { "UIDS": ["rWrCdr321Qu", "..."] },
    "ORG_UNIT_GROUPS": { "cOK4Feyi0nP": ["cOK4Feyi0nP"] }
}
```

`extract_config.json` sets the extraction window/mode and the data elements to pull, grouped into named extracts:

```json
{
    "SETTINGS": {
        "SOURCE_DHIS2_CONNECTION": "drc-snis",
        "NUMBER_MONTHS_WINDOW": 4,
        "CMM_WINDOW_MONTHS": 6,
        "MODE": "DOWNLOAD_REPLACE"
    },
    "ORG_UNIT_GROUPS": { "OUG_URBAN": "cOK4Feyi0nP" },
    "DATA_ELEMENTS": {
        "EXTRACTS": [
            { "EXTRACT_UID": "fosa_morbidity", "UIDS": ["aZwnLALknnj", "..."], "ORG_UNITS_LEVEL": 5, "FREQUENCY": "MONTHLY" }
        ]
    },
    "REPORTING_RATES": { "EXTRACTS": [] },
    "INDICATORS": { "EXTRACTS": [] }
}
```

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `start_date` | string | Auto | Start of the extraction window in `YYYYMM` format. If not provided, defaults to `SETTINGS.STARTDATE` in `extract_config.json`, or current date minus `SETTINGS.NUMBER_MONTHS_WINDOW` months if that is also unset. |
| `end_date` | string | Auto | End of the extraction window in `YYYYMM` format. If not provided, defaults to `SETTINGS.ENDDATE` in `extract_config.json`, or current date minus 1 month if that is also unset. |
| `run_metadata_data` | bool | `true` | Extract the org unit pyramid and org unit groups from the source DHIS2. |
| `run_extract_data` | bool | `true` | Extract morbidity data elements from the source DHIS2. |
| `skip_existing_extracts` | bool | `false` | If `true`, skips downloading extracts that already exist on disk. |
| `add_to_dataset` | bool | `true` | Push all extracts produced in this run to the OpenHEXA dataset. |

> Resolved `start_date`/`end_date` are clamped to a minimum of `201701`, and the run fails if the start date ends up after the end date.

## Output dataset

Extracts are added as a new version to the **`snis-prs-cmm-extract`** dataset (version prefix: `PRS_DS_CMM_extract`).

## Output files

All files are saved in **Parquet** format.

- **Org unit pyramid** (`pyramid_data.parquet`) — organisation units up to level 5, restricted to the selected provinces (`sync_config.json`) and all their descendants.
- **Org unit groups** (`org_unit_groups.parquet`) — the urban Zones de Santé org unit group (`extract_config.json`).
- **Data element extracts** — one file per month (e.g. `data_{YYYYMM}.parquet`) per extract configuration (currently `fosa_morbidity`), covering the requested window extended backward by the configured CMM window (default 6 months).
- **Mapping file** (`updates_collector.json`) — a JSON file included in every dataset version that maps each extract file name to its extract identifier, used by downstream integration processes.
