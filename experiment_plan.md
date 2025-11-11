# 6-Hour Study & Experiment Plan: Comparing Chunking Techniques

This plan helps you implement and compare Fixed, Embedding-based, and Dual Semantic Chunker (DSC) methods for procedural knowledge extraction (as in your thesis IPKE project).

---

## Hour 1: Prep & Understanding
- Skim key papers/docs for each method (Fixed, Embedding-based, DSC).
- Diagram chunking on a sample paragraph.
- Summarize the main differences and the metrics you'll use.
- **Checkpoint:** Write 4-5 bullet points summarizing your understanding.

## Hour 2: Set Up Baseline (Fixed Chunking)
- Implement/review fixed-size chunking in the IPKE pipeline.
- Run on 1–2 sample docs.
- Save and note extraction output issues.
- **Question:** What broke with fixed chunking?

## Hour 3: Implement Embedding-Based Chunking
- Use SBERT or SentenceTransformers for sentence embeddings.
- Code grouping of semantically similar sentences into chunks (use cosine similarity, e.g. 0.7 threshold).
- Run pipeline with semantic chunking.
- Compare with fixed-size output; log 2-3 improvements/issues.

## Hour 4: Dual Semantic Chunker (DSC) Prototyping
- Implement DSC: sentence boundaries + similarity aggregation (threshold e.g. 0.75).
- Run DSC chunking and inspect outputs.
- **Check-in:** Did DSC better preserve multi-sentence context? Give 2–3 cases.

## Hour 5: Evaluation & Metrics
- Use Step F1, Adjacency F1, Constraint Attachment, Macro A-score.
- Adapt/write code to score all three chunkings on test docs.
- Make table/bar chart of results.
- Analyze metric patterns.
- **Checkpoint:** Note the most surprising result.

## Hour 6: Interpretation & Synthesis
- Write a comparison paragraph: strengths, weaknesses, and metric results for each method.
- Identify use cases where each chunking is best/worst.
- Draft 2–3 slides (diag. + findings for thesis or sharing).
- **Final reflection:** What challenge is next? Write 1–2 open questions.

---

## Good luck! Let me know if you want sample code or metric scripts. Save this as `experiment_plan.md` or similar for easy reference.