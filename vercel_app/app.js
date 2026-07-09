const fileInput = document.getElementById("file");
const drop = document.getElementById("drop");
const dropText = document.getElementById("dropText");
const runBtn = document.getElementById("run");
const results = document.getElementById("results");
const statusEl = document.getElementById("status");
const csvBtn = document.getElementById("csv");

let dataUri = null;
let lastFeatures = [];

function setStatus(msg, isErr = false) {
  statusEl.textContent = msg;
  statusEl.className = "status" + (isErr ? " err" : "");
}

fileInput.addEventListener("change", () => {
  const f = fileInput.files[0];
  if (!f) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    dataUri = e.target.result;
    dropText.textContent = "✅ " + f.name;
    runBtn.disabled = false;
    setStatus("");
  };
  reader.readAsDataURL(f);
});

["dragover", "dragenter"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("hover"); })
);
["dragleave", "drop"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("hover"); })
);
drop.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files[0];
  if (f) { fileInput.files = e.dataTransfer.files; fileInput.dispatchEvent(new Event("change")); }
});

runBtn.addEventListener("click", async () => {
  if (!dataUri) return;
  runBtn.disabled = true;
  setStatus("Segmenting… (first run may take a few seconds to warm up)");
  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: dataUri }),
    });
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || "Request failed");

    document.getElementById("inImg").src = data.input;
    document.getElementById("outImg").src = data.overlay;

    const s = data.stats;
    document.getElementById("metrics").innerHTML = [
      ["Cells detected", s.cell_count],
      ["Mean area (px)", s.mean_area],
      ["Mean eccentricity", s.mean_eccentricity],
      ["Mean intensity", s.mean_intensity],
    ].map(([k, v]) => `<div class="metric"><div class="v">${v}</div><div class="k">${k}</div></div>`).join("");

    lastFeatures = data.features || [];
    csvBtn.hidden = lastFeatures.length === 0;
    results.hidden = false;
    setStatus(`Done — ${s.cell_count} nuclei detected.`);
  } catch (err) {
    setStatus("Error: " + err.message, true);
  } finally {
    runBtn.disabled = false;
  }
});

csvBtn.addEventListener("click", () => {
  if (!lastFeatures.length) return;
  const cols = Object.keys(lastFeatures[0]);
  const rows = [cols.join(",")].concat(
    lastFeatures.map((r) => cols.map((c) => r[c]).join(","))
  );
  const blob = new Blob([rows.join("\n")], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "cellsight_features.csv";
  a.click();
});
