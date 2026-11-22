const $ = (id) => document.getElementById(id);
let cases = [];
let selected = null;
let detailRequest = 0;
let deliveryEnabled = false;

async function request(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    const failure = new Error(typeof error.detail === "string" ? error.detail : "Request failed.");
    failure.code = error.code;
    throw failure;
  }
  return response.json();
}

function state(entry) {
  if (entry.review) return "reviewed";
  if (entry.action === "escalate" || (entry.status === "failed" && entry.attempts >= 3)) return "human";
  if (entry.action === "reply") return "draft";
  return entry.status;
}

const labels = {
  reviewed: "Reviewed", draft: "Draft ready", human: "Needs human",
  failed: "Waiting to retry", processing: "Processing", processed: "Processed"
};

function renderQueue() {
  $("queue").replaceChildren();
  const filter = $("filter").value;
  const visible = cases.filter((entry) => filter === "all" || state(entry) === filter);
  for (const entry of visible) {
    const button = document.createElement("button");
    button.className = "case-row";
    button.setAttribute("aria-pressed", String(selected === entry.id));
    button.textContent = entry.subject || "No subject";
    const meta = document.createElement("small");
    meta.textContent = [labels[state(entry)], entry.section].filter(Boolean).join(" · ");
    button.append(meta);
    button.addEventListener("click", () => openCase(entry.id).catch(showError));
    $("queue").append(button);
  }
  if (!visible.length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "No matching cases.";
    $("queue").append(empty);
  }
}

async function loadQueue() {
  cases = await request("/api/queue");
  $("total").textContent = cases.length;
  $("pending").textContent = cases.filter((entry) => state(entry) === "draft").length;
  $("human").textContent = cases.filter((entry) => state(entry) === "human").length;
  renderQueue();
}

async function openCase(id) {
  const sequence = ++detailRequest;
  const entry = await request("/api/queue/" + id);
  if (sequence !== detailRequest) return;
  selected = id;
  renderQueue();
  $("subject").textContent = entry.subject || "No subject";
  $("reason").textContent = entry.reason || entry.error || labels[state(entry)];
  $("body").textContent = entry.body || "Message body unavailable.";
  $("case-content").hidden = false;
  $("draft-content").hidden = entry.action !== "reply";
  $("trace").textContent = JSON.stringify({prompt_version: entry.prompt_version || "unrecorded", trace_id: entry.trace_id, stages: entry.stages, sources: entry.sources, delivery: entry.delivery_status}, null, 2);
  $("send").hidden = !deliveryEnabled || entry.review !== "approve" || Boolean(entry.delivery_status);
  $("draft").value = entry.draft || "";
  $("draft").readOnly = Boolean(entry.review);
  $("review-info").textContent = entry.review
    ? (entry.review === "approve" ? "Approved" : "Rejected") + " by " + entry.reviewer
    : "Approval saves this exact reply for a separate send action.";
  $("approve").hidden = Boolean(entry.review);
  $("reject").hidden = Boolean(entry.review);
}

function showError(error) {
  if (error.code === "gmail-reauthorize") {
    $("gmail-warning").textContent = error.message;
    $("gmail-warning").hidden = false;
    $("status").textContent = "";
    return;
  }
  $("status").textContent = error.message;
}

function post(url, data) {
  return request(url, {
    method: "POST",
    headers: {"Content-Type": "application/json", "X-Pingo-Action": "dashboard"},
    body: JSON.stringify(data)
  });
}

async function review(action) {
  const id = selected;
  $("approve").disabled = $("reject").disabled = true;
  try {
    await post("/api/queue/" + id + "/review", {action, draft: $("draft").value});
    await loadQueue();
    if (selected === id) await openCase(id);
    $("status").textContent = "Review saved.";
  } catch (error) {
    showError(error);
  } finally {
    $("approve").disabled = $("reject").disabled = false;
  }
}

$("approve").addEventListener("click", () => review("approve"));
$("reject").addEventListener("click", () => review("reject"));
$("filter").addEventListener("change", renderQueue);
$("poll").addEventListener("click", async () => {
  $("poll").disabled = true;
  $("status").textContent = "Checking Gmail…";
  try {
    const result = await post("/api/poll", {});
    $("gmail-warning").hidden = true;
    await loadQueue();
    $("status").textContent = result.failed ? "Inbox check finished with a processing error."
      : result.found ? "Inbox check complete." : "No unprocessed emails found.";
  } catch (error) {
    showError(error);
  } finally {
    $("poll").disabled = false;
  }
});
request("/api/status").then((status) => {
  deliveryEnabled = status.delivery_enabled === true;
  $("delivery-mode").textContent = deliveryEnabled ? "Delivery enabled" : "Delivery disabled";
  if (selected !== null) return openCase(selected);
}).catch(showError);
loadQueue().catch(showError);


$("send").addEventListener("click", async () => {
  if (!window.confirm("Send this approved reply to the customer through Gmail?")) return;
  $("send").disabled = true;
  try {
    await post("/api/queue/" + selected + "/send", {});
    await openCase(selected);
    $("status").textContent = "Approved reply sent.";
  } catch (error) {
    showError(error);
  } finally {
    $("send").disabled = false;
  }
});


async function openResources(kind) {
  const entries = await request("/api/" + kind);
  $("resources").hidden = false;
  $("resource-title").textContent = kind === "knowledge" ? "Knowledge base" : "Evaluation runs";
  $("resource-body").replaceChildren();
  if (!entries.length) $("resource-body").textContent = "No saved records yet.";
  for (const entry of entries) {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = kind === "knowledge" ? entry.section + " · " + entry.title
      : "Run " + entry.id + " · " + new Date(entry.created_at).toLocaleString();
    details.append(summary);
    if (kind === "knowledge") {
      const input = document.createElement("textarea");
      input.value = entry.content;
      input.maxLength = 20000;
      const save = document.createElement("button");
      save.textContent = "Save article";
      save.addEventListener("click", async () => {
        save.disabled = true;
        try {
          await request("/api/knowledge/" + entry.id, {
            method: "PUT",
            headers: {"Content-Type": "application/json", "X-Pingo-Action": "dashboard"},
            body: JSON.stringify({content: input.value, previous_content: entry.content})
          });
          entry.content = input.value.trim();
          $("status").textContent = "Article saved with edit history.";
        } catch (error) {
          showError(error);
        } finally {
          save.disabled = false;
        }
      });
      details.append(input, save);
    } else {
      const pre = document.createElement("pre");
      pre.className = "message";
      pre.textContent = JSON.stringify(entry.report, null, 2);
      details.append(pre);
    }
    $("resource-body").append(details);
  }
}
$("knowledge-nav").addEventListener("click", () => openResources("knowledge").catch(showError));
$("eval-nav").addEventListener("click", () => openResources("evaluations").catch(showError));
$("resource-close").addEventListener("click", () => { $("resources").hidden = true; });
