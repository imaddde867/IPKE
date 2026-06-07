# ECIR paper replaces thesis evaluation numbers entirely

The thesis reports Φ = 0.362 for DSC on 3 documents (3M SOP, DOA Food,
op_firesafety) using the heuristic parent-boundary implementation.

The ECIR paper will evaluate on 8 reviewed-gold documents using the global DP
implementation (ADR-0001). Both the algorithm and the eval set change simultaneously.

We do not re-report thesis numbers in the ECIR paper. The ECIR paper cites the
thesis as prior work and states: "We extend and correct the evaluation." All
claims are grounded in the 8-document reviewed-gold results with multi-seed CIs.

Reporting both sets of numbers would require a controlled comparison
(same algorithm, different eval set) that is not the contribution of the paper,
and would force an explanation of why the numbers differ without a clean causal story.
