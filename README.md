# DRC SNIS Data Repository

This OpenHEXA workspace acts as a **dedicated SNIS data repository** for the Democratic Republic of Congo. Its  purpose is to extract health data from the DRC SNIS DHIS2 instance and make it available to other workspaces — it does not perform any analysis or transformation. All pipelines in this repository follow the same pattern: connect to SNIS, extract, store as Parquet files, and publish to a shared OpenHEXA dataset.

Downstream workspaces consume these datasets for data integration or data analytics processes.

---

## Data integration pipelines

### DHIS2 SNIS Extract

Extracts the core SNIS routine health data on a monthly basis, including routine data elements, reporting rates, and ACM indicators, along with the org unit pyramid and population figures.

| | |
|---|---|
| **Source** | DHIS2 SNIS (`drc-snis`) |
| **Data types** | Data elements, indicators, reporting rates |
| **Storage path** | `pipelines/dhis2_snis_extract/data` |
| **Format** | `.parquet` |
| **Dataset** | [SNIS Extracts dataset](https://app.openhexa.org/workspaces/drc-dsnis/datasets/snis-extracts/from/drc-dsnis/) |
| **Shared to** | DRC PNLP |

---

### DHIS2 SNIS Data Elements Sync

Extracts a specific set of raw data elements from SNIS, scoped to periods from January 2025 onward. Complements the main extract for data elements that require a separate extraction flow.

| | |
|---|---|
| **Source** | DHIS2 SNIS (`drc-snis`) |
| **Data types** | Data elements |
| **Storage path** | `pipelines/dhis2_snis_data_elements_extract/data` |
| **Format** | `.parquet` |
| **Dataset** | [SNIS Data elements extracts dataset](https://app.openhexa.org/workspaces/drc-dsnis/datasets/snis-data-elements-extracts/from/drc-dsnis/) |
| **Shared to** | DRC PNLP |

---

### DHIS2 SNIS Sentinel Sites

Extracts data from SNIS sentinel site datasets. Each sentinel dataset is extracted independently and published alongside a mapping file that links each extract file to its source dataset identifier.

| | |
|---|---|
| **Source** | DHIS2 SNIS (`drc-snis`) |
| **Data types** | Data elements |
| **Storage path** | `pipelines/dhis2_snis_sentinel_extract/data` |
| **Format** | `.parquet` |
| **Dataset** | [SNIS Sentinel dataset](https://app.openhexa.org/workspaces/drc-dsnis/datasets/snis-sentinel-dataset/from/drc-dsnis/) |
| **Shared to** | DRC PNLP |

---

### DHIS2 SNIS Palu Data Mensuel

Compiles monthly malaria data by combining data elements and reporting rates extracted directly from SNIS with data sourced from the SNIS Extract pipeline outputs. Only periods from January 2025 onward are supported.

| | |
|---|---|
| **Source** | DHIS2 SNIS (`drc-snis`) + SNIS Extract pipeline outputs |
| **Data types** | Data elements, reporting rates, population|
| **Storage path** | `pipelines/dhis2_snis_palu_data_mensuel/data` |
| **Format** | `.parquet` |
| **Dataset** | [`SNIS Palu mensuel extracts`](https://app.openhexa.org/workspaces/drc-dsnis/datasets/snis-palu-mensuel-extracts/from/drc-dsnis/)|
| **Shared to** | RDC Palu rapports mensuels (`rdc-palu-rapports-mensuels`) |

---

### DHIS2 PRS Dataset Extract

Extracts a subset of SNIS data scoped for the PRS integration: data elements at facility and zone de santé levels, restricted to a configured list of provinces, along with the corresponding org unit pyramid.

| | |
|---|---|
| **Source** | DHIS2 SNIS (`drc-snis`) |
| **Data types** | Data elements, org unit pyramid |
| **Storage path** | `pipelines/dhis2_prs_dataset_extract/data` |
| **Format** | `.parquet` |
| **Dataset** | [SNIS PRS dataset sync](https://app.openhexa.org/workspaces/drc-dsnis/datasets/snis-prs-dataset-sync/from/drc-dsnis/) |
| **Shared to** | DRC PRS (`drc-prs`) |

---

### DHIS2 SNIS PRS CMM Morbidity Extract

Extracts morbidity data elements from SNIS for all FOSA (health facilities) under a configured list of 20 provinces (sync_config.json), used to compute the CMM (Consommation Moyenne Mensuelle) for the PRS integration. Extraction periods are extended backwards by a configurable CMM window (default 6 months) to support the rolling average calculation, along with the org unit pyramid and the urban Zones de Santé org unit group.

| | |
|---|---|
| **Source** | DHIS2 SNIS (`drc-snis`) |
| **Data types** | Data elements (morbidity), org unit pyramid, org unit groups |
| **Storage path** | `pipelines/dhis2_snis_prs_cmm_morbidity_extract/data` |
| **Format** | `.parquet` |
| **Dataset** | [SNIS PRS CMM extract dataset](https://app.openhexa.org/workspaces/drc-dsnis/datasets/snis-prs-cmm-extract/from/drc-dsnis/) |
| **Shared to** | DRC PRS (`drc-prs`) |
