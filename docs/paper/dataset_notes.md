# Dataset Notes

These notes describe the public dataset slice for IPKE issue 53 and the ECIR paper.

## Final Source Families

- `USGS` - public field sampling and groundwater technical procedures.
- `FAA` - aviation maintenance technician handbooks.
- `EPA` - SOP guidance and field procedure documents.
- `CDC_NIOSH` - industrial hygiene and analytical methods.
- `NASA` - procedural directives and safety requirements.
- `OLSK` - Open Lab Starter Kit Small CNC workbooks and support files.

## Tier-A Selected Documents

Initial Tier-A gold annotation targets:

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

Downloaded alternates and support artifacts are tracked in
`datasets/paper/public_sources_manifest.csv`.

## IAA Plan

Use a planned inter-annotator agreement subset before freezing the Tier-A gold set:

- Sample at least 20 percent of Tier-A documents, with a minimum of three source families.
- Include one government SOP-like document, one maintenance or assembly document, and one safety or requirements document.
- Double-annotate step boundaries, step order, constraint attachment, resource references, and safety-critical flags.
- Resolve disagreements before final gold freeze and record adjudication notes outside raw source files.

## Limitations

- The first wave is public and reproducible, but it is not a substitute for private partner SOPs.
- Several sources are long handbooks with mixed procedural and expository content, so section selection must be documented.
- NASA and EPA directives can be policy-heavy; they should not dominate step-level evaluation.
- Open-hardware OLSK documents improve assembly coverage but are smaller than government manuals.
- The public corpus is English-only in this slice.
- Raw PDFs are not committed, so downstream reproduction depends on stable source URLs and manifest hashes.

## Exclusions

MyFixit and iFixit-derived content are excluded because their repair instructions are attractive
for procedural extraction but carry licensing and platform-terms risk for AI/ML use without
explicit permission. Do not use them in the paper corpus unless permission is secured and recorded.

## Rights Screen

- USGS-authored material is generally U.S. Public Domain; screen embedded third-party material.
- FAA documents are official U.S. federal sources with no CC license shown on the portal; screen embedded third-party material.
- EPA documents are official U.S. federal sources with no CC license shown on cited pages; screen embedded third-party material.
- CDC/NIOSH material follows CDC public-domain reuse with attribution and no-endorsement unless otherwise noted.
- NASA procedural directives require document-by-document rights screening.
- OLSK manuals and media are CC BY-SA 4.0; hardware design files are CERN-OHL-W-2.0.

Rights notes must travel with extracted text, gold files, and experiment metadata. Do not publish
raw PDFs from this repository.

