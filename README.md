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
| **Dataset** | `snis-palu-mensuel-extracts` |
| **Shared to** | RDC Palu rapports mensuels (`rdc-palu-rapports-mensuels`) |
