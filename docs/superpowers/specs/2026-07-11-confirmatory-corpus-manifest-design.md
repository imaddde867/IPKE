# Confirmatory Corpus Manifest Design

## Decision

Use one typed JSON manifest as the machine-readable source of corpus membership for IPKE
evaluation. Directory presence, `review_status`, and the source-download manifest do not
define experiment membership.

This implements the frozen-split requirement in ADR-0005 and the method-paper design,
plus issue #112's exclusion decision. The manifest is supporting research
infrastructure, not a benchmark artifact.

## Alternatives considered

1. **Move excluded JSON files to another directory.** Simple at runtime, but rewrites the
   historical layout, couples scientific status to filesystem location, and makes audit
   history harder to follow.
2. **Use a plain text list of included document IDs.** Small, but cannot state why a file
   is excluded, distinguish a requirements stress test from a repair candidate, or
   validate corpus state.
3. **Use a typed JSON manifest. Chosen.** Slightly more structure, but it records every
   candidate exactly once, makes exclusions reviewable, and gives validation and
   evaluation one shared selection contract.

## Manifest contract

Create `datasets/paper/corpus_manifest.json`:

```json
{
  "schema_version": 1,
  "manifest_status": "provisional",
  "documents": [
    {
      "doc_id": "epa_guidance_preparing_sops_qag6",
      "source_family": "epa",
      "role": "procedure_candidate",
      "status": "candidate",
      "include_for_evaluation": true,
      "reason": "Manual source audit and human verification pending."
    },
    {
      "doc_id": "nasa_npr_8715_3d_general_safety",
      "source_family": "nasa",
      "role": "requirements_stress_test",
      "status": "excluded_wrong_genre",
      "include_for_evaluation": false,
      "reason": "Section 2.5.2 is a requirements block; see issue #112."
    }
  ]
}
```

Allowed values:

- `manifest_status`: `provisional` or `frozen`;
- `role`: `procedure_candidate` or `requirements_stress_test`;
- `status`: `candidate`, `excluded_wrong_genre`, or
  `excluded_pending_reannotation`.

Consistency rules:

- document IDs are non-empty and unique;
- every JSON stem in the selected gold directory appears exactly once;
- every manifest ID resolves to one JSON file;
- included entries have role `procedure_candidate` and status `candidate`;
- excluded entries use an excluded status and carry a non-empty reason;
- at least one entry is included;
- a paper gate requires `manifest_status = "frozen"`;
- development dry runs may use a provisional manifest but must report that status.

The first manifest includes the five unaudited or partially corrected legacy procedure
candidates. NASA, OLSK, and NIOSH are classified but excluded. Inclusion does not make
the five candidates paper eligible; the separate structural, grounding, human, and
split gates still apply.

## Code boundary

Create `src/evaluation/corpus_manifest.py` with Pydantic v2 boundary models and two
public functions:

```python
def load_corpus_manifest(path: Path) -> CorpusManifest: ...

def select_manifest_gold_files(
    manifest: CorpusManifest,
    gold_dir: Path,
) -> tuple[Path, ...]: ...
```

The loader owns JSON shape and cross-entry consistency. File selection owns the exact
one-to-one check against a gold directory. Neither function changes annotations.

## Consumer behavior

`scripts/validate_paper_gold.py` gains:

- `--manifest PATH`, which selects only included files but verifies that all directory
  JSON files are classified;
- `--require-frozen-manifest`, which fails closed unless the supplied manifest is
  frozen.

`make eval-paper-gate` supplies both flags plus `--strict` and
`--require-human-verified`. At the current state it must fail because the manifest is
provisional and the five included files are unsigned. It must not fail on excluded NASA,
OLSK, or NIOSH annotations.

`scripts/eval_multiseed.py` gains `--manifest PATH`. It filters gold/text pairs before
loading annotation JSON or starting configuration/model work. The Make `eval` and
`eval-full` targets supply the same manifest. `eval-full` remains dependent on
`eval-paper-gate`, so a provisional or unsigned corpus cannot start a paper sweep.

The existing `--allow-unverified` development override remains limited to human evidence;
it does not alter manifest inclusion. A developer may run included provisional
candidates directly, but such outputs remain development-only.

## Failure behavior

Manifest parse, schema, duplicate, missing-file, extra-file, empty-inclusion, or
consistency errors are printed clearly and return exit 1 before model work. Excluded
files are preserved on disk and remain available for historical diagnostics, manual
repair, or a separate stress-test protocol.

## Tests

- current manifest classifies all eight legacy files and selects exactly five;
- NASA is machine-readable as `requirements_stress_test`;
- OLSK and NIOSH are excluded pending reannotation;
- duplicate, missing, extra, inconsistent, and empty manifests fail;
- paper gate rejects provisional status and the five unsigned included files without
  naming the three excluded files as evidence failures;
- runner dry-run counts only included files;
- excluded malformed annotation JSON is not loaded by a manifest-scoped runner;
- full evaluation still depends on the paper gate and passes the manifest explicitly.

## Non-goals

- No annotation label is created or changed.
- No historical file is moved or deleted.
- No final development/test source-family split is chosen in this change.
- No OLSK, NIOSH, or NASA representation is rebuilt.
- No result is regenerated or reinterpreted.
