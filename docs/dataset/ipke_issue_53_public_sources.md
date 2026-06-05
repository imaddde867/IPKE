# IPKE Issue 53 Public Sources

This note records the first-wave public source selection for the ECIR paper dataset. The
machine-readable source of truth is `datasets/paper/public_sources_manifest.csv`.

## Source Selection

The initial corpus uses official government and open-hardware documents because they are
stable, procedure-dense, and suitable for reproducible public experiments without partner data.

Final source families:

- `USGS` - water sampling and groundwater technical procedures.
- `FAA` - aviation maintenance technician handbooks.
- `EPA` - SOP guidance, field sampling, and procedure validation documents.
- `CDC_NIOSH` - analytical and industrial hygiene method documents.
- `NASA` - procedural directives and safety requirements.
- `OLSK` - Open Lab Starter Kit Small CNC workbooks and support artifacts.

## Tier-A Gold Targets

The first Tier-A annotation wave is the set of rows with `selected_for_gold=true`:

- `usgs_nfm_collection_water_samples_a4`
- `usgs_groundwater_technical_procedures_tm1_a1`
- `faa_amt_general_handbook_2023`
- `faa_amt_airframe_handbook`
- `epa_guidance_preparing_sops_qag6`
- `epa_field_sampling_measurement_procedure_validation`
- `epa_field_operations_manual_filter_sampling_sop`
- `niosh_nmam_5th_edition_ebook`
- `nasa_npr_8715_3d_general_safety`
- `olsk_small_cnc_v1_workbook`

The remaining downloaded sources are alternates or support artifacts. They can support
resource grounding, future annotations, or ablations, but should not silently enter the Tier-A
gold set without updating the manifest and paper notes.

## Downloaded Corpus

The first-wave raw corpus is kept in an ignored local raw-document directory generated from
`datasets/paper/public_sources_manifest.csv`.

Only filenames, SHA-256 hashes, and byte sizes are committed in the manifest. Raw PDFs and
spreadsheets must stay outside git.

## Usage And Rights Notes

- USGS-authored material is generally U.S. Public Domain; screen embedded third-party material.
- FAA documents are official U.S. federal sources with no CC license shown on the portal; screen embedded third-party material.
- EPA documents are official U.S. federal sources with no CC license shown on cited pages; screen embedded third-party material.
- CDC/NIOSH material follows CDC public-domain reuse with attribution and no-endorsement unless otherwise noted.
- NASA procedural directives require document-by-document rights screening.
- OLSK Small CNC manuals and media are CC BY-SA 4.0; hardware design files are CERN-OHL-W-2.0.

Rights notes are dataset-screening notes, not legal advice. Before publishing redistributed text
snippets, check each extracted section for embedded third-party images, tables, vendor marks, or
non-federal content.

## Excluded Sources

MyFixit and iFixit-derived material are excluded from the paper corpus because repair-step content
is useful but carries licensing and platform-terms risk for AI/ML use without explicit permission.
They should not be used for training, annotation transfer, or evaluation unless permission is
obtained and recorded.
