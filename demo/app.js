const $ = (id) => document.getElementById(id);
function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}
function chip(item, select) {
  const button = el(
    "button",
    `${item.id} · ${item.text || item.label}`,
    item.text ? `constraint ${item.type}` : "step",
  );
  button.dataset.item = item.id;
  button.onclick = () => select(item);
  return button;
}

async function load() {
  const response = await fetch("data/epa.json");
  if (!response.ok)
    throw new Error(`Snapshot request failed (${response.status})`);
  const data = await response.json();
  const { annotation, constraints, legacy, source, review_packet: packet } = data;
  $("title").textContent = annotation.procedure.title;
  $("counts").textContent =
    `${annotation.steps.length} steps · ${constraints.length} constraints · ${packet.human_decisions_required.length} open human decisions`;
  $("replay").textContent = data.replay.reason;
  const chars = Array.from(source.text); // Python offsets count code points, not UTF-16 units.
  function select(item, reveal = true) {
    if (!item) {
      $("inspector").replaceChildren(
        el("p", "This reference does not resolve in the snapshot.", "status"),
      );
      return;
    }
    const p = item.provenance || {};
    const exactSpan =
      Number.isInteger(p.char_start) && Number.isInteger(p.char_end);
    for (const node of document.querySelectorAll("[data-item]")) {
      node.setAttribute(
        "aria-pressed",
        String(exactSpan && node.dataset.item === item.id),
      );
    }
    const targets = item.attached_to || item.applies_to;
    const applicable = annotation.steps.includes(item)
      ? annotation.constraints.filter((c) => c.applies_to.includes(item.id))
      : [];
    const details = [
      el("h3", `${item.id} · ${item.type || "step"}`),
      el("p", item.text || item.label),
      el(
        "p",
        targets
          ? `Enforcement: ${item.enforcement} · Applies to: ${targets.join(", ")}`
          : "Step in proposed source order",
      ),
    ];
    if (applicable.length)
      details.push(
        el(
          "p",
          `Procedure-level constraint: ${applicable.map((c) => `${c.id} · ${c.text}`).join("; ")}`,
        ),
      );
    details.push(
      el(
        "p",
        exactSpan
          ? `Section ${p.section} · page ${p.page} · [${p.char_start}, ${p.char_end})`
          : `Section ${p.section ?? "unrecorded"} · page ${p.page ?? "unrecorded"} · section-level provenance only`,
        "muted",
      ),
      el(
        "p",
        exactSpan
          ? "Agent proposal · human verification pending"
          : "Legacy candidate · unverified · no exact item span",
        "status",
      ),
    );
    $("inspector").replaceChildren(...details);
    if (!exactSpan) {
      // Legacy items only record a section; never invent an exact source span.
      $("source").replaceChildren(document.createTextNode(chars.join("")));
      $("source").scrollTop = 0;
    } else {
      const a = p.char_start - source.char_start,
        b = p.char_end - source.char_start;
      const mark = el("mark", chars.slice(a, b).join(""));
      $("source").replaceChildren(
        document.createTextNode(chars.slice(0, a).join("")),
        mark,
        document.createTextNode(chars.slice(b).join("")),
      );
      // Scroll only the source pane; keep keyboard focus on the selected node.
      $("source").scrollTop = Math.max(0, mark.offsetTop - 80);
    }
    if (reveal && matchMedia("(max-width: 850px)").matches) {
      $("inspector").closest("aside").scrollIntoView({ block: "start" });
    }
  }
  for (const step of annotation.steps) {
    const row = el("div", undefined, "graph-row");
    row.append(chip(step, select));
    const lane = el("div", undefined, "constraint-lane");
    for (const c of step.constraints) lane.append(chip(c, select));
    if (!step.constraints.length) {
      const applicable = annotation.constraints.filter((c) =>
        c.applies_to.includes(step.id),
      );
      lane.append(
        el(
          "span",
          applicable.length
            ? `No step-local constraint; procedure-level ${applicable.map((c) => c.id).join(", ")} applies`
            : "No constraint applies to this step in this proposal",
          "muted",
        ),
      );
    }
    row.append(lane);
    $("graph").append(row);
  }
  const globals = el("section", undefined, "global-constraints");
  globals.append(el("h3", "Procedure-level constraints"));
  for (const c of annotation.constraints) {
    globals.append(
      chip(c, select),
      el("p", `Applies to ${c.applies_to.join(", ")}`, "muted"),
    );
  }
  $("graph").append(globals);
  const omitted = data.summary.omitted_constraint_ids;
  const omittedDetails = el("details");
  omittedDetails.append(
    el("summary", "Inspect each omitted clause"),
    el(
      "p",
      "This constructed example illustrates information loss; it does not measure model summary quality.",
      "muted",
    ),
  );
  for (const c of constraints.filter((c) => omitted.includes(c.id)))
    omittedDetails.append(chip(c, select));
  $("summary").append(
    el("p", "Authored illustration · not a recorded model output", "status"),
    el("p", data.summary.text, "summary-text"),
    el(
      "h3",
      `${omitted.length} specific constraints omitted from this illustration`,
    ),
    omittedDetails,
  );
  function representation(summary) {
    $("graph").hidden = summary;
    $("summary").hidden = !summary;
    $("representation").textContent = summary
      ? `${omitted.length} constraints omitted from the paragraph; expand the omitted-clause list to inspect their source spans.`
      : `${constraints.length} explicit constraints shown with their attachments. Spine order follows the candidate NEXT relations.`;
    $("summary-toggle").textContent = summary
      ? "Restore procedure graph"
      : "Show illustrative summary";
    $("summary-toggle").setAttribute("aria-pressed", String(summary));
  }
  $("summary-toggle").onclick = () => representation($("summary").hidden);
  representation(false);
  for (const button of document.querySelectorAll("[data-view]")) {
    button.onclick = () => {
      for (const other of document.querySelectorAll("[data-view]")) {
        const active = button === other;
        other.setAttribute("aria-pressed", String(active));
        $(`${other.dataset.view}-view`).hidden = !active;
      }
    };
  }
  $("audit-counts").textContent =
    `Legacy candidate: ${packet.legacy_candidate.step_count} steps / ${packet.legacy_candidate.constraint_count} constraints → Proposed: ${packet.review_candidate.step_count} steps / ${packet.review_candidate.constraint_count} constraints. Status: ${packet.status}.`;
  const refItems = new Map();
  for (const s of annotation.steps) refItems.set(`review:${s.id}`, s);
  for (const c of constraints) refItems.set(`review:${c.id}`, c);
  for (const s of legacy.steps) {
    refItems.set(`legacy:${s.id}`, s);
    // Legacy constraints carry no provenance; inherit their step's section only.
    for (const c of s.constraints)
      refItems.set(`legacy:${c.id}`, { ...c, provenance: s.provenance });
  }
  for (const c of legacy.constraints) refItems.set(`legacy:${c.id}`, c);
  for (const kind of ["step", "constraint"]) {
    $("decisions").append(el("h3", `Proposed ${kind} decisions`));
    for (const d of packet[`proposed_${kind}_decisions`]) {
      const card = el("article", undefined, "decision");
      card.append(
        el("strong", d.action.toUpperCase()),
        el(
          "p",
          `${d.input_refs.join(", ") || "∅"} → ${d.output_refs.join(", ") || "∅"}`,
        ),
        el("p", d.rationale),
      );
      for (const ref of [...d.input_refs, ...d.output_refs]) {
        const button = el("button", `Inspect ${ref} source`);
        button.onclick = () => {
          document.querySelector('[data-view="graph"]').click();
          representation(false);
          select(refItems.get(ref));
        };
        card.append(button);
      }
      $("decisions").append(card);
    }
  }
  for (const h of packet.human_decisions_required)
    $("human-decisions").append(
      el("p", `${h.decision_id} · OPEN · ${h.question}`, "decision"),
    );
  $("packet").textContent = JSON.stringify(packet, null, 2);
  $("legacy").textContent = JSON.stringify(data.legacy, null, 2);
  const corpus = data.corpus_summary;
  $("corpus-status").textContent =
    `${corpus.human_verified} of ${corpus.documents} corpus documents are human verified.`;
  for (const doc of data.corpus) {
    const card = el("article", undefined, "decision");
    const link = el("a", "Original source ↗");
    link.href = doc.source_url;
    card.append(
      el("h3", doc.title),
      el("p", doc.status, "status"),
      el("p", doc.reason),
      el("p", doc.usage, "muted"),
      link,
    );
    $("corpus").append(card);
  }
  $("loading").hidden = true;
  $("console").hidden = false;
  select(constraints[0], false);
}
load().catch((error) => {
  $("loading").textContent =
    `Unable to load the snapshot. Serve demo/ over HTTP and check data/epa.json. ${error.message}`;
});
