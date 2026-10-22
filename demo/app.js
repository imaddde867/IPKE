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
  const { annotation, constraints, source, review_packet: packet } = data;
  $("title").textContent = annotation.procedure.title;
  $("counts").textContent =
    `${annotation.steps.length} steps · ${constraints.length} constraints · ${packet.human_decisions_required.length} open human decisions`;
  $("replay").textContent = data.replay.reason;
  const chars = Array.from(source.text); // Python offsets count code points, not UTF-16 units.
  function select(item, reveal = true) {
    for (const node of document.querySelectorAll("[data-item]")) {
      node.setAttribute("aria-pressed", String(node.dataset.item === item.id));
    }
    const p = item.provenance;
    const targets = item.attached_to || item.applies_to;
    $("inspector").replaceChildren(
      el("h3", `${item.id} · ${item.type || "step"}`),
      el("p", item.text || item.label),
      el(
        "p",
        targets
          ? `Enforcement: ${item.enforcement} · Applies to: ${targets.join(", ")}`
          : "Step in proposed source order",
      ),
      el(
        "p",
        `Section ${p.section} · page ${p.page} · [${p.char_start}, ${p.char_end})`,
        "muted",
      ),
      el("p", "Agent proposal · human verification pending", "status"),
    );
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
    if (reveal && matchMedia("(max-width: 850px)").matches) {
      $("inspector").closest("aside").scrollIntoView({ block: "start" });
    }
  }
  for (const step of annotation.steps) {
    const row = el("div", undefined, "graph-row");
    row.append(chip(step, select));
    const lane = el("div", undefined, "constraint-lane");
    for (const c of step.constraints) lane.append(chip(c, select));
    if (!step.constraints.length)
      lane.append(
        el("span", "No attached constraint in this proposal", "muted"),
      );
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
  $("summary").append(
    el("p", "Authored illustration · not a recorded model output", "status"),
    el("p", data.summary.text, "summary-text"),
    el(
      "h3",
      `${omitted.length} specific constraints omitted from this illustration`,
    ),
    el(
      "p",
      "Inspect each omitted clause below. This constructed example illustrates information loss; it does not measure model summary quality.",
    ),
  );
  for (const c of constraints.filter((c) => omitted.includes(c.id)))
    $("summary").append(chip(c, select));
  function representation(summary) {
    $("graph").hidden = summary;
    $("summary").hidden = !summary;
    $("representation").textContent = summary
      ? `${omitted.length} constraints omitted from the paragraph; their source clauses remain inspectable below.`
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
      for (const ref of d.output_refs) {
        const id = ref.split(":")[1];
        const item = [...annotation.steps, ...constraints].find(
          (i) => i.id === id,
        );
        const button = el("button", `Inspect ${ref} source`);
        button.onclick = () => {
          document.querySelector('[data-view="graph"]').click();
          representation(false);
          select(item);
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
