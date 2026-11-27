// หน้าบ้าน: เลือกไฟล์ → ตรวจล่วงหน้า → ยืนยันนำเข้า → แสดงผล → บันทึกผลตรวจ
// ฟังก์ชันลูกศรใน map/filter/forEach ใช้แปลงหรือเลือกข้อมูลตามฟังก์ชันหลักที่ครอบอยู่
// $ ค้นองค์ประกอบด้วย id; esc ป้องกันข้อความข้อมูลถูกตีความเป็น HTML
const $ = (id) => document.getElementById(id),
  esc = (v) =>
    String(v ?? "").replace(
      /[&<>"']/g,
      (c) =>
        ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;",
        }[c])
    );
// จัดรูปแบบจำนวนและสัดส่วนสำหรับแสดงผลภาษาไทย
const num = (n) => Number(n).toLocaleString("th-TH"),
  pct = (n) => (n * 100).toFixed(2) + "%";
let dataset = "",
  page = 1,
  pages = 1,
  selectedFile = null,
  listRequest = 0,
  detailRequest = 0;
// ส่งคำขอ HTTP อ่าน JSON และเปลี่ยนข้อผิดพลาดจากหลังบ้านเป็น Error
async function api(path, options) {
  const r = await fetch(path, options);
  const d = await r.json();
  if (!r.ok) throw Error(d.error || "เชื่อมต่อไม่สำเร็จ");
  return d;
}
// แสดงข้อความผิดพลาดในแถบแจ้งเตือนของหน้าเว็บ
function error(e) {
  $("error").textContent = e.message;
  $("error").hidden = false;
}
// สลับหน้าจอและเมนู; ถ้ายังไม่เลือกข้อมูลจะพากลับหน้านำเข้า
function view(id) {
  if (!dataset && ["overview", "ranking"].includes(id)) id = "upload";
  if (!["overview", "ranking", "upload", "model"].includes(id)) id = "overview";
  document.querySelectorAll(".view").forEach((e) => (e.hidden = e.id !== id));
  document
    .querySelectorAll("nav [data-view]")
    .forEach((e) => e.classList.toggle("active", e.dataset.view === id));
  $("breadcrumb").textContent = {
    overview: "ศูนย์ปฏิบัติงาน",
    ranking: "รายการตรวจสอบ",
    upload: "วิเคราะห์ข้อมูลใหม่",
    model: "ข้อมูลโมเดล",
  }[id];
  if (id === "ranking") loadList();
}
document
  .querySelectorAll("[data-view]")
  .forEach((e) => e.addEventListener("click", () => view(e.dataset.view)));
document
  .querySelector(".brand")
  .addEventListener("click", () => view("overview"));
// สร้าง HTML ของคะแนนพร้อมแถบความยาวตามคะแนน ไม่ใช่เปอร์เซ็นต์ความมั่นใจ
function score(s) {
  return `<div class="score">${s.toFixed(
    4
  )}<span class="score-track"><i style="width:${s * 100}%"></i></span></div>`;
}
// สร้างแถวตารางมิเตอร์และปุ่มตรวจสอบ โดย escape ข้อความจากข้อมูลก่อนใส่ HTML
function rows(items, full = false) {
  return items.length
    ? items
        .map(
          (r) =>
            `<tr><td>${num(r.rank)}</td><td class="meter-id">${esc(
              r.id
            )}</td><td>${score(r.score)}</td>${
              full ? `<td>${pct(r.missing)}</td>` : ""
            }<td><span class="status ${r.alert ? "alert" : ""}">${
              r.alert ? "ควรตรวจสอบ" : "ต่ำกว่าเกณฑ์"
            }</span></td><td><span class="work-badge ${esc(r.work_status)}">${
              workLabel[r.work_status]
            }</span><small class="officer-name">${esc(
              r.officer || "ยังไม่มีผู้บันทึก"
            )}</small></td><td><button class="text-btn" data-meter="${esc(
              r.id
            )}">เปิดตรวจสอบ →</button></td></tr>`
        )
        .join("")
    : `<tr><td colspan="${
        full ? 7 : 6
      }" class="empty">ไม่พบมิเตอร์ตามเงื่อนไข</td></tr>`;
}
// เรียกสรุปชุดข้อมูลและอัปเดตการ์ด สัดส่วนความเสี่ยง จำนวนที่ข้าม และคิวงาน
async function refresh() {
  if (!dataset) {
    $("datasetBadge").textContent = "ยังไม่ได้เลือกข้อมูล";
    $("loading").hidden = true;
    $("topk").textContent = "เลือกชุดข้อมูลเพื่อดูข้อมูลประกอบการวางแผน";
    return;
  }
  $("loading").hidden = false;
  $("error").hidden = true;
  try {
    const s = await api("/api/summary?dataset=" + dataset);
    $("datasetBadge").textContent = s.name;
    $("datasetSelect").value = dataset;
    $("workTotal").textContent = num(s.work.total);
    $("workPending").textContent = num(s.work.pending);
    $("workProgress").textContent = num(s.work.in_progress);
    $("workDone").textContent = num(s.work.completed);
    $("scopeNote").textContent =
      "แสดงผลคัดกรองของชุดข้อมูลที่เลือก จำนวนผู้มีความเสี่ยงไม่ใช่จำนวนผู้ยืนยันว่าลักลอบใช้ไฟ";
    $("topk").innerHTML = s.topk?.length
      ? s.topk
          .map(
            (t) =>
              `<div class="topk-row"><span>Top ${
                t.k
              }</span><div class="bar-track"><i style="width:${
                (t.found / t.k) * 100
              }%"></i></div><strong>${pct(t.found / t.k)}</strong><small>พบ ${
                t.found
              } จาก ${t.k} ราย</small></div>`
          )
          .join("")
      : '<p class="empty">ต้องมี FLAG และข้อมูลอย่างน้อย 100 ราย เพื่อแสดง Precision@K</p>';
    const fraction = s.count ? s.alerts / s.count : 0,
      below = s.count - s.alerts;
    $("riskTotal").textContent = num(s.count);
    $("riskAlerts").textContent = num(s.alerts);
    $("riskPercent").textContent = pct(fraction);
    $("riskBelow").textContent = num(below);
    $("belowPercent").textContent =
      pct(s.count ? below / s.count : 0) + " ของทั้งหมด";
    $("riskSummary").textContent = `จากผู้ใช้ไฟ ${num(
      s.count
    )} ราย พบผู้ที่มีความเสี่ยงตามเกณฑ์โมเดล ${num(s.alerts)} ราย คิดเป็น ${pct(
      fraction
    )}`;
    if (s.skipped_count) {
      $("riskSummary").textContent = `ไฟล์ต้นฉบับ ${num(s.source_count)} ราย · ข้ามข้อมูลไม่เพียงพอ ${num(s.skipped_count)} ราย · ` + $("riskSummary").textContent;
    }
    $("riskSegment").style.width = fraction * 100 + "%";
    $("riskLegend").textContent = `${num(s.alerts)} ราย (${pct(fraction)})`;
    $("belowLegend").textContent = `${num(below)} ราย (${pct(
      s.count ? below / s.count : 0
    )})`;
    $("riskDistribution").setAttribute(
      "aria-label",
      $("riskSummary").textContent
    );
    $(
      "riskThreshold"
    ).textContent = `คัดกรองด้วย XGBoost ที่ Threshold ${s.threshold.toFixed(
      4
    )} · เปอร์เซ็นต์นี้คือสัดส่วนจำนวนรายที่ถึงเกณฑ์ ไม่ใช่โอกาสลักลอบรายบุคคล · ต่ำกว่าเกณฑ์ไม่ได้ยืนยันว่าไม่มีการลักลอบ`;
  } catch (e) {
    error(e);
  } finally {
    $("loading").hidden = true;
  }
}
// รวมตัวกรองและเลขหน้าปัจจุบันเป็น query string สำหรับรายการและการส่งออก
function query() {
  return new URLSearchParams({
    dataset,
    search: $("search").value,
    status: $("status").value,
    work: $("workFilter").value,
    page,
  });
}
// โหลดรายการมิเตอร์ตามตัวกรอง; ใช้เลขคำขอป้องกันผลเก่ามาเขียนทับผลใหม่
async function loadList() {
  const request = ++listRequest;
  try {
    const r = await api("/api/meters?" + query());
    if (request !== listRequest) return;
    $("rankingRows").innerHTML = rows(r.rows, true);
    pages = r.pages;
    $("resultCount").textContent = num(r.count) + " ราย";
    $("pageLabel").textContent = `หน้า ${page} / ${num(pages)}`;
    $("prev").disabled = page <= 1;
    $("next").disabled = page >= pages;
  } catch (e) {
    error(e);
  }
}
let timer;
// หน่วงการค้นหาเล็กน้อยก่อนโหลดรายการ เพื่อลดคำขอระหว่างพิมพ์
$("search").addEventListener("input", () => {
  clearTimeout(timer);
  timer = setTimeout(() => {
    page = 1;
    loadList();
  }, 250);
});
// เมื่อเปลี่ยนผลคัดกรอง ให้เริ่มรายการที่หน้าแรก
$("status").addEventListener("change", () => {
  page = 1;
  loadList();
});
// เปิดหน้ารายการก่อนหน้า
$("prev").onclick = () => {
  page--;
  loadList();
};
// เปิดหน้ารายการถัดไป
$("next").onclick = () => {
  page++;
  loadList();
};
// ดาวน์โหลด CSV ตามตัวกรองปัจจุบัน
$("export").onclick = () => {
  location.href = "/api/export?" + query();
};
let uploadRevision = 0,
  validatedFile = null,
  duplicateFiles = [],
  insufficientCount = 0,
  eligibleCount = 0,
  uploadBusy = false;
// เปิดปุ่มนำเข้าเฉพาะไฟล์ที่ตรวจแล้ว มีรายที่ใช้ได้ ตั้งชื่อแล้ว และยืนยันการข้ามหรือไฟล์ซ้ำครบ
function updateImportButton() {
  $("analyze").disabled =
    uploadBusy ||
    eligibleCount === 0 ||
    (insufficientCount > 0 && !$("skipInsufficient").checked) ||
    validatedFile !== selectedFile ||
    !selectedFile ||
    !$("datasetName").value.trim() ||
    (duplicateFiles.length > 0 && !$("allowDuplicate").checked);
}
// รับไฟล์ใหม่ ตั้งชื่อเริ่มต้น และล้างผลตรวจหรือคำยืนยันของไฟล์เก่า
function selectFile(f) {
  if (uploadBusy) return;
  uploadRevision++;
  selectedFile = f;
  validatedFile = null;
  duplicateFiles = [];
  insufficientCount = 0;
  eligibleCount = 0;
  $("insufficientConsent").hidden = true;
  $("skipInsufficient").checked = false;
  $("filename").textContent = f
    ? `${f.name} · ${(f.size / 1024).toFixed(0)} KB`
    : "ยังไม่ได้เลือกไฟล์";
  $("datasetName").value = f ? f.name.replace(/\.csv$/i, "").slice(0, 120) : "";
  $("validateFile").disabled = !f;
  $("uploadPreview").hidden = true;
  $("duplicateConsent").hidden = true;
  $("allowDuplicate").checked = false;
  $("uploadProgress").textContent = "";
  updateImportButton();
}
// ตรวจสิทธิ์เปิดปุ่มนำเข้าใหม่เมื่อผู้ใช้ยืนยันข้ามรายข้อมูลไม่เพียงพอ
$("skipInsufficient").onchange = updateImportButton;
// ตรวจชื่อชุดข้อมูลเมื่อผู้ใช้พิมพ์
$("datasetName").oninput = updateImportButton;
// ตรวจสิทธิ์เปิดปุ่มเมื่อยืนยันนำเข้าไฟล์ซ้ำ
$("allowDuplicate").onchange = updateImportButton;
// รับไฟล์จากช่องเลือกแล้วส่งไป selectFile
$("file").onchange = (e) => selectFile(e.target.files[0]);
["dragover", "dragleave", "drop"].forEach((type) =>
  $("dropzone").addEventListener(type, (e) => {
    e.preventDefault();
    $("dropzone").classList.toggle("drag", type === "dragover");
    if (type === "drop") selectFile(e.dataTransfer.files[0]);
  })
);
// ส่งไฟล์ไปตรวจล่วงหน้า แสดงคุณภาพข้อมูล รายที่ข้ามได้ และไฟล์ซ้ำ โดยยังไม่นำเข้า
$("validateFile").onclick = async () => {
  const f = selectedFile,
    revision = uploadRevision;
  if (!f) return;
  validatedFile = null;
  updateImportButton();
  $("validateFile").disabled = true;
  $("uploadProgress").textContent = "กำลังตรวจสอบไฟล์…";
  $("error").hidden = true;
  try {
    if (!/\.csv$/i.test(f.name)) throw Error("กรุณาเลือกไฟล์ CSV");
    if (f.size > 200 * 1024 * 1024) throw Error("ไฟล์ต้องไม่เกิน 200 MB");
    const r = await api("/api/upload/preview", {
      method: "POST",
      headers: { "Content-Type": "text/csv" },
      body: f,
    });
    if (revision !== uploadRevision) return;
    validatedFile = f;
    duplicateFiles = r.duplicates;
    insufficientCount = r.insufficient_count;
    eligibleCount = r.eligible_count;
    $("skipInsufficient").checked = false;
    $("insufficientConsent").hidden = !insufficientCount || !eligibleCount;
    $("allowDuplicate").checked = false;
    $("duplicateConsent").hidden = !duplicateFiles.length;
    $("uploadPreview").hidden = false;
    $("uploadPreview").innerHTML = `<h3>${insufficientCount ? "พบรายที่ข้อมูลไม่เพียงพอ" : "ตรวจสอบรูปแบบผ่าน"}</h3><p>${num(
      r.count
    )} ราย · ${num(r.days)} วัน<br>${esc(r.start)} ถึง ${esc(r.end)}<br>${
      r.has_labels ? "มีผลจริง FLAG" : "ไม่มีผลจริง FLAG"
    }</p><p>ช่องข้อมูลขาดหรือค่าที่ใช้ไม่ได้ ${num(r.missing)} ช่อง (${
      r.missing_percent
    }%)</p>${
      r.missing
        ? '<p class="import-warning">ค่าว่าง ค่าที่ไม่ใช่ตัวเลข ค่าติดลบ และค่าไม่จำกัดจะถือเป็นข้อมูลขาด และจัดการตามขั้นตอนของโมเดล</p>'
        : ""
    }${
      duplicateFiles.length
        ? '<p class="import-warning"><strong>พบไฟล์เนื้อหาเหมือนกันทุกไบต์ที่เคยนำเข้าแล้ว</strong></p>' +
          duplicateFiles
            .map(
              (d, i) =>
                `<button type="button" data-existing="${i}">เปิดชุดเดิม: ${esc(
                  d.name
                )}</button>`
            )
            .join("")
        : "<p>ไม่พบไฟล์ซ้ำในประวัติการนำเข้า</p>"
    }`;
    if (insufficientCount) {
      $("uploadPreview").insertAdjacentHTML("beforeend", `
        <div class="import-warning">
          <strong>ข้อมูลไม่เพียงพอ ${num(insufficientCount)} ราย · วิเคราะห์ได้ ${num(eligibleCount)} ราย</strong>
          <p>รายเหล่านี้ไม่มีค่าการใช้ไฟที่ใช้ได้เลย จะไม่ได้รับคะแนนและไม่จัดเป็นกลุ่มต่ำกว่าเกณฑ์</p>
          <details><summary>ดูรหัสมิเตอร์ที่ข้อมูลไม่เพียงพอ</summary>
            <div style="max-height:220px;overflow:auto;overflow-wrap:anywhere">${r.insufficient_ids.map(id => `<div>${esc(id)}</div>`).join("")}</div>
          </details>
          ${!eligibleCount ? "<p>ไม่สามารถนำเข้าได้ กรุณาใช้ไฟล์อื่น</p>" : ""}
        </div>`);
    }
    $("uploadPreview")
      .querySelectorAll("[data-existing]")
      .forEach(
        (b) =>
          // เปิดชุดข้อมูลเดิมจากคำเตือนไฟล์ซ้ำ แทนการสร้างชุดใหม่
          (b.onclick = async () => {
            dataset = duplicateFiles[Number(b.dataset.existing)].token;
            try {
              await loadDatasets();
              await refresh();
              view("overview");
            } catch (e) {
              error(e);
            }
          })
      );
    $("uploadProgress").textContent = "ตรวจชื่อชุดข้อมูล แล้วกดยืนยันนำเข้า";
  } catch (e) {
    if (revision === uploadRevision) {
      error(e);
      $("uploadPreview").hidden = true;
      $("duplicateConsent").hidden = true;
      $("uploadProgress").textContent =
        "ยังไม่นำเข้าข้อมูล กรุณาแก้ไขไฟล์แล้วตรวจสอบใหม่";
    }
  } finally {
    if (revision === uploadRevision) {
      $("validateFile").disabled = false;
      updateImportButton();
    }
  }
};
// ส่งไฟล์พร้อมชื่อและคำยืนยันไปวิเคราะห์ แล้วโหลดภาพรวมของชุดข้อมูลใหม่
$("analyze").onclick = async () => {
  updateImportButton();
  if ($("analyze").disabled) return;
  uploadBusy = true;
  $("file").disabled = true;
  $("datasetName").disabled = true;
  $("validateFile").disabled = true;
  updateImportButton();
  $("error").hidden = true;
  $("uploadProgress").textContent =
    "กำลังสร้าง Features และวิเคราะห์ด้วย XGBoost…";
  try {
    const q = new URLSearchParams({
      name: $("datasetName").value.trim(),
      allow_duplicate: String($("allowDuplicate").checked),
      skip_insufficient: String($("skipInsufficient").checked),
    });
    const r = await api("/api/upload?" + q, {
      method: "POST",
      headers: { "Content-Type": "text/csv" },
      body: selectedFile,
    });
    dataset = r.dataset;
    validatedFile = null;
    $("search").value = "";
    $("status").value = "all";
    await loadDatasets();
    await refresh();
    $("uploadProgress").textContent = `นำเข้าและวิเคราะห์แล้ว ${num(
      r.count
    )} ราย`;
    view("overview");
  } catch (e) {
    error(e);
    validatedFile = null;
    $("uploadProgress").textContent =
      "นำเข้าไม่สำเร็จ กรุณาตรวจสอบไฟล์อีกครั้ง";
  } finally {
    uploadBusy = false;
    $("file").disabled = false;
    $("datasetName").disabled = false;
    $("validateFile").disabled = false;
    updateImportButton();
  }
};

// สร้างกราฟ SVG รายวันและค่าเฉลี่ย พร้อมเว้นเส้นเมื่อข้อมูลขาด
function chart(d) {
  const w = 900,
    h = 250,
    left = 60,
    right = 20,
    top = 15,
    bottom = 35;
  const max = Math.max(
    1,
    ...d.usage.filter((v) => v !== null),
    ...d.average.filter((v) => v !== null)
  );
  // แปลงลำดับวันและค่าการใช้ไฟเป็นพิกัดบน SVG
  const x = (i) => left + (i / (d.usage.length - 1)) * (w - left - right),
    y = (v) => h - bottom - (v / max) * (h - top - bottom);
  // แปลงค่ารายวันเป็นเส้น SVG โดยเริ่มเส้นใหม่หลังช่วงข้อมูลขาด
  function path(values) {
    let pen = false;
    return values
      .map((v, i) => {
        if (v === null) {
          pen = false;
          return "";
        }
        const s = (pen ? "L" : "M") + x(i).toFixed(1) + "," + y(v).toFixed(1);
        pen = true;
        return s;
      })
      .join(" ");
  }
  let grid = "";
  for (let i = 0; i <= 4; i++) {
    const v = (max * i) / 4;
    grid += `<line x1="${left}" x2="${w - right}" y1="${y(v)}" y2="${y(
      v
    )}" stroke="#e5ebee"/><text x="${left - 8}" y="${
      y(v) + 4
    }" text-anchor="end" fill="#6d7f8a" font-size="11">${v.toFixed(0)}</text>`;
  }
  [0, Math.floor((d.dates.length - 1) / 2), d.dates.length - 1].forEach((i) => {
    grid += `<text x="${x(i)}" y="${h - 8}" text-anchor="${
      i === 0 ? "start" : i === d.dates.length - 1 ? "end" : "middle"
    }" fill="#6d7f8a" font-size="11">${esc(d.dates[i])}</text>`;
  });
  return `<svg class="usage-svg" viewBox="0 0 ${w} ${h}" role="img" aria-label="กราฟปริมาณการใช้ไฟรายวันและค่าเฉลี่ย 30 วัน ช่องว่างคือวันที่ข้อมูลขาด">${grid}${
    d.usage.length <= 100
      ? d.usage
          .map((v, i) =>
            v === null
              ? ""
              : `<circle cx="${x(i)}" cy="${y(
                  v
                )}" r="3" fill="#3379bb"><title>${esc(d.dates[i])}: ${v.toFixed(
                  2
                )} kWh</title></circle>`
          )
          .join("")
      : ""
  }<path d="${path(
    d.usage
  )}" stroke="#99c6bc" stroke-width="1" fill="none"/><path d="${path(
    d.average
  )}" stroke="#107e5e" stroke-width="2" fill="none"/></svg>`;
}
// โหลดรายละเอียดมิเตอร์และแสดงหน้าต่างกราฟ SHAP กับแบบฟอร์มผลตรวจ
async function detail(id) {
  const request = ++detailRequest;
  $("detailBody").innerHTML =
    '<p class="loading">กำลังโหลดกราฟการใช้ไฟและคำนวณ SHAP…</p>';
  if (!$("detailDialog").open) $("detailDialog").showModal();
  try {
    const d = await api("/api/meter?" + new URLSearchParams({ id, dataset }));
    if (request !== detailRequest) return;
    const max = Math.max(...d.reasons.map((r) => Math.abs(r.impact)), 0.001);
    $("detailBody").innerHTML = `<p class="detail-id">${esc(
      d.id
    )}</p><div class="detail-stats"><div><span>คะแนนโมเดล</span><strong>${d.score.toFixed(
      4
    )}</strong></div><div><span>สถานะ</span><strong>${
      d.alert ? "ควรตรวจสอบ" : "ต่ำกว่าเกณฑ์"
    }</strong></div><div><span>ข้อมูลขาด</span><strong>${pct(
      d.missing
    )}</strong></div></div><div id="inspectionPanel"></div><h3>ประวัติการใช้ไฟรายวัน</h3><div class="chart-tools"><button type="button" class="secondary" id="zoomIn">＋ ซูมเข้า</button><button type="button" class="secondary" id="zoomOut">− ซูมออก</button><button type="button" class="secondary" id="chartReset">ดูทั้งหมด</button><button type="button" class="secondary" id="chartPrev">← ช่วงก่อนหน้า</button><button type="button" class="secondary" id="chartNext">ช่วงถัดไป →</button></div><div class="chart-dates"><label>ตั้งแต่<input type="date" id="chartStart"></label><label>ถึง<input type="date" id="chartEnd"></label><button type="button" class="secondary" id="chartApply">แสดงช่วงวันที่</button></div><p id="chartRange" role="status" class="muted"></p><div class="chart-frame" id="usageChart">${chart(
      d
    )}</div><label class="chart-pan">เลื่อนช่วงเวลา<input type="range" id="chartPan" min="0" value="0" step="1"></label><p id="chartPoint" class="footnote">ชี้หรือแตะบนกราฟเพื่อดูค่ารายวัน · ใช้ Ctrl + ล้อเมาส์เพื่อซูม</p><div class="chart-legend"><span><i class="legend-dot"></i>ค่ารายวัน</span><span><i class="legend-dot average"></i>เฉลี่ยย้อนหลัง 30 วัน</span></div><p class="footnote">ช่องว่างในเส้นรายวันหมายถึงไม่มีข้อมูล ไม่ได้แทนด้วยศูนย์ · เส้นเฉลี่ยใช้ค่าที่มีอย่างน้อย 10 วันในช่วง 30 วัน</p><h3>ปัจจัยที่มีผลต่อคะแนน (SHAP)</h3><p class="muted">สีส้มเพิ่มคะแนน · สีฟ้าลดคะแนน · หน่วย log-odds</p>${d.reasons
      .map(
        (r) =>
          `<div class="shap-row"><span>${esc(r.label)}<small>${esc(
            r.feature
          )} · ค่า ${
            r.value === null
              ? "ไม่มีข้อมูล"
              : r.value.toLocaleString("th-TH", { maximumFractionDigits: 3 })
          }</small></span><div class="shap-track"><i class="${
            r.impact < 0 ? "negative" : ""
          }" style="left:${
            r.impact < 0 ? 50 - (Math.abs(r.impact) / max) * 50 : 50
          }%;width:${(Math.abs(r.impact) / max) * 50}%"></i></div><strong>${
            r.impact > 0 ? "+" : ""
          }${r.impact.toFixed(3)}</strong></div>`
      )
      .join(
        ""
      )}<p class="notice">คะแนนผ่านเกณฑ์ ${(0.29896414279937744).toFixed(
      4
    )} เป็นสัญญาณให้ตรวจสอบเพิ่มเติม ไม่ใช่คำยืนยันการลักลอบ${
      d.truth !== null
        ? ` · ป้ายกำกับเดิมในข้อมูล: ${
            d.truth === 1 ? "ลักลอบ (1)" : "ปกติ (0)"
          }`
        : ""
    }</p>`;
    renderInspection(d.id, d.inspection);
    setupChart(d);
  } catch (e) {
    $("detailBody").textContent = e.message;
  }
}
// รับการคลิกปุ่มมิเตอร์จากตารางที่สร้างใหม่ แล้วเปิดรายละเอียด
document.addEventListener("click", (e) => {
  const button = e.target.closest("[data-meter]");
  if (button) detail(button.dataset.meter);
});
// ปิดรายละเอียดและยกเลิกการนำผลคำขอเก่ามาแสดง
$("closeDetail").onclick = () => {
  $("detailDialog").close();
  detailRequest++;
};
// โหลดรายการหน้าแรกตามสถานะงานใหม่
$("workFilter").onchange = () => {
  page = 1;
  loadList();
};
// เปิดรายการที่ถึงเกณฑ์ความเสี่ยงโดยไม่จำกัดสถานะงาน
$("openQueue").onclick = () => {
  $("workFilter").value = "all";
  $("status").value = "alert";
  $("search").value = "";
  page = 1;
  view("ranking");
};

const workLabel = {
  pending: "รอตรวจสอบ",
  in_progress: "กำลังดำเนินการ",
  completed: "ตรวจแล้ว",
};
const outcomeLabel = {
  "": "ยังไม่มีผลตรวจ",
  found: "พบการลักลอบ",
  not_found: "ไม่พบการลักลอบ",
  inconclusive: "ยังสรุปไม่ได้",
};
// โหลดชื่อชุดข้อมูลที่บันทึกไว้เพื่อสร้างตัวเลือกในเมนู
async function loadDatasets() {
  const list = await api("/api/datasets");
  $("datasetSelect").innerHTML =
    '<option value="">เลือกชุดข้อมูลเพื่อเริ่มงาน</option>' +
    list
      .map(
        (d) =>
          `<option value="${esc(d.token)}">${esc(d.name)} · ${new Date(
            d.created
          ).toLocaleString("th-TH")} · ${d.token.slice(0, 6)}</option>`
      )
      .join("");
  $("datasetSelect").value = dataset;
}
// เปลี่ยนชุดข้อมูล ล้างการค้นหา และโหลดภาพรวมใหม่
$("datasetSelect").onchange = async () => {
  dataset = $("datasetSelect").value;
  page = 1;
  $("search").value = "";
  await refresh();
  view("overview");
};
document.querySelectorAll("[data-work]").forEach(
  (b) =>
    // ใช้การ์ดสถานะงานเป็นทางลัดไปยังรายการที่กรองแล้ว
    (b.onclick = () => {
      $("search").value = "";
      $("workFilter").value = b.dataset.work;
      $("status").value = b.dataset.work === "pending" ? "alert" : "all";
      page = 1;
      view("ranking");
    })
);
// สร้างแบบฟอร์มผลตรวจและประวัติ พร้อมผูกการส่งบันทึกไป API
function renderInspection(id, inspection) {
  const r = inspection.record;
  $(
    "inspectionPanel"
  ).innerHTML = `<section class="inspection-card"><div class="panel-heading"><div><p class="eyebrow">ผลตรวจของเจ้าหน้าที่</p><h3>บันทึกการดำเนินงาน</h3></div><span class="work-badge ${esc(
    r.work_status
  )}">${workLabel[r.work_status]}</span></div><p class="footnote">${
    dataset === "test" ? "บันทึกสำหรับชุดสาธิต SGCC เท่านั้น · " : ""
  }ชื่อผู้บันทึกเป็นข้อมูลที่กรอกเอง ยังไม่ได้ยืนยันผ่านบัญชีผู้ใช้</p><form id="inspectionForm"><div class="inspection-fields"><label>สถานะงาน<select name="work_status">${Object.entries(
    workLabel
  )
    .map(
      ([v, t]) =>
        `<option value="${v}" ${
          r.work_status === v ? "selected" : ""
        }>${t}</option>`
    )
    .join(
      ""
    )}</select></label><label>ชื่อผู้บันทึก<input name="officer" required maxlength="100" value="${esc(
    r.officer
  )}" placeholder="ชื่อเจ้าหน้าที่"></label><label>ผลตรวจยืนยัน<select name="outcome"><option value="">เลือกผลตรวจ</option>${Object.entries(
    outcomeLabel
  )
    .filter(([v]) => v)
    .map(
      ([v, t]) =>
        `<option value="${v}" ${
          r.outcome === v ? "selected" : ""
        }>${t}</option>`
    )
    .join(
      ""
    )}</select></label><label>วันที่ตรวจ<input name="inspected_on" type="date" value="${esc(
    r.inspected_on
  )}"></label></div><label>หมายเหตุ<textarea name="notes" rows="3" maxlength="5000" placeholder="รายละเอียดที่ตรวจพบหรือสิ่งที่ต้องติดตาม">${esc(
    r.notes
  )}</textarea></label><div class="save-row"><button class="primary" type="submit">บันทึกการตรวจสอบ</button><span id="saveStatus" role="status"></span></div></form><details><summary>ประวัติการบันทึก (${
    inspection.history.length
  } ครั้ง)</summary><div class="inspection-history">${
    inspection.history.length
      ? inspection.history
          .map(
            (h) =>
              `<article><strong>${workLabel[h.work_status]} · ${
                outcomeLabel[h.outcome]
              }</strong><p>${esc(h.officer)} · ${new Date(
                h.updated_at
              ).toLocaleString("th-TH")}${
                h.inspected_on ? " · วันที่ตรวจ " + esc(h.inspected_on) : ""
              }</p><p class="history-notes">${esc(
                h.notes || "ไม่มีหมายเหตุ"
              )}</p></article>`
          )
          .join("")
      : "<p>ยังไม่มีการบันทึก</p>"
  }</div></details></section>`;
  const form = $("inspectionForm");
  // เปิดและกำหนดช่องผลตรวจกับวันที่เป็นช่องบังคับเฉพาะสถานะตรวจแล้ว
  function sync() {
    const complete = form.elements.work_status.value === "completed";
    for (const name of ["outcome", "inspected_on"]) {
      form.elements[name].disabled = !complete;
      form.elements[name].required = complete;
      if (!complete) form.elements[name].value = "";
    }
  }
  form.elements.work_status.onchange = sync;
  sync();
  // ส่งผลตรวจกับ version ปัจจุบัน เพื่อให้หลังบ้านตรวจการแก้ไขชนกันก่อนบันทึก
  form.onsubmit = async (e) => {
    e.preventDefault();
    const button = form.querySelector("button");
    button.disabled = true;
    $("saveStatus").textContent = "กำลังบันทึก…";
    const currentDataset = dataset;
    try {
      const fields = Object.fromEntries(new FormData(form));
      fields.outcome = fields.outcome || "";
      fields.inspected_on = fields.inspected_on || "";
      const result = await api("/api/inspection", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...fields,
          id,
          dataset: currentDataset,
          version: r.version,
        }),
      });
      if (dataset !== currentDataset) return;
      renderInspection(id, result);
      $("saveStatus").textContent = "บันทึกเรียบร้อย";
      await refresh();
      if (!$("ranking").hidden) await loadList();
    } catch (err) {
      $("saveStatus").textContent = err.message;
      button.disabled = false;
    }
  };
}
// เริ่มที่หน้านำเข้า โหลดเฉพาะรายการชุดข้อมูลที่มีอยู่ ไม่โหลดชุดสาธิต
view("upload");
$("loading").hidden = true;
loadDatasets().catch(error);

// เตรียมการซูม เลื่อน เลือกช่วงวันที่ และชี้ดูค่าบนกราฟ
function setupChart(d) {
  const count = d.dates.length,
    minimum = Math.min(7, count);
  let start = 0,
    size = count;
  const host = $("usageChart");
  // วาดกราฟใหม่ในช่วงที่เลือกและปรับสถานะปุ่ม ขอบเขตวันที่ และแถบเลื่อน
  function render() {
    start = Math.max(0, Math.min(Math.round(start), count - size));
    const end = start + size;
    host.innerHTML = chart({
      dates: d.dates.slice(start, end),
      usage: d.usage.slice(start, end),
      average: d.average.slice(start, end),
    });
    $("chartStart").value = d.dates[start];
    $("chartEnd").value = d.dates[end - 1];
    for (const id of ["chartStart", "chartEnd"]) {
      $(id).min = d.dates[0];
      $(id).max = d.dates[count - 1];
    }
    $("chartRange").textContent = `${d.dates[start]} ถึง ${
      d.dates[end - 1]
    } · ${num(size)} วัน · แกนตั้งปรับตามช่วงที่เลือก (kWh)`;
    $("chartPan").max = count - size;
    $("chartPan").value = start;
    $("chartPan").disabled = size === count;
    $("chartPrev").disabled = start === 0;
    $("chartNext").disabled = end === count;
    $("zoomIn").disabled = size <= minimum;
    $("zoomOut").disabled = size === count;
    $("chartPoint").textContent =
      "ชี้หรือแตะบนกราฟเพื่อดูค่ารายวัน · ใช้ Ctrl + ล้อเมาส์เพื่อซูม";
  }
  // เปลี่ยนจำนวนวันที่แสดง โดยคงตำแหน่งจุดยึดและจำกัดช่วงซูม
  function zoom(factor, anchor = 0.5) {
    const next = Math.max(minimum, Math.min(count, Math.round(size * factor)));
    start += (size - next) * anchor;
    size = next;
    render();
  }
  // ซูมเข้าโดยลดจำนวนวันที่แสดงครึ่งหนึ่ง
  $("zoomIn").onclick = () => zoom(0.5);
  // ซูมออกโดยเพิ่มจำนวนวันที่แสดงสองเท่า
  $("zoomOut").onclick = () => zoom(2);
  // คืนกราฟให้แสดงทุกวัน
  $("chartReset").onclick = () => {
    start = 0;
    size = count;
    render();
  };
  // เลื่อนกราฟไปช่วงเวลาก่อนหน้า
  $("chartPrev").onclick = () => {
    start -= Math.max(1, Math.floor(size / 2));
    render();
  };
  // เลื่อนกราฟไปช่วงเวลาถัดไป
  $("chartNext").onclick = () => {
    start += Math.max(1, Math.floor(size / 2));
    render();
  };
  // เลื่อนจุดเริ่มต้นของช่วงกราฟตามแถบเลื่อน
  $("chartPan").oninput = (e) => {
    start = Number(e.target.value);
    render();
  };
  // ตรวจช่วงวันที่ที่เลือกก่อนปรับกราฟ
  $("chartApply").onclick = () => {
    const a = $("chartStart").value,
      b = $("chartEnd").value;
    const first = d.dates.findIndex((v) => v >= a);
    let last = -1;
    d.dates.forEach((v, i) => {
      if (v <= b) last = i;
    });
    if (!a || !b || a > b || first < 0 || last - first + 1 < minimum) {
      $(
        "chartRange"
      ).textContent = `เลือกช่วงวันที่ในข้อมูลอย่างน้อย ${minimum} วัน`;
      return;
    }
    start = first;
    size = last - first + 1;
    render();
  };
  // แปลงตำแหน่งเมาส์บนกราฟเป็นสัดส่วน 0–1 สำหรับซูมและเลือกวัน
  function fraction(e) {
    const rect = host.querySelector("svg").getBoundingClientRect();
    return Math.max(
      0,
      Math.min(1, (((e.clientX - rect.left) / rect.width) * 900 - 60) / 820)
    );
  }
  // ใช้ Ctrl พร้อมล้อเมาส์เพื่อซูมกราฟรอบตำแหน่งเมาส์
  host.addEventListener(
    "wheel",
    (e) => {
      if (!e.ctrlKey) return;
      e.preventDefault();
      zoom(e.deltaY < 0 ? 0.8 : 1.25, fraction(e));
    },
    { passive: false }
  );
  // แสดงค่ารายวันและค่าเฉลี่ยของวันที่ตรงกับตำแหน่งที่ชี้หรือแตะ
  host.onpointermove = host.onpointerdown = (e) => {
    const i = start + Math.round(fraction(e) * (size - 1));
    // จัดข้อความค่าที่ชี้และแสดงกรณีไม่มีข้อมูลอย่างชัดเจน
    const value = (v) =>
      v === null
        ? "ไม่มีข้อมูล"
        : v.toLocaleString("th-TH", { maximumFractionDigits: 2 }) + " kWh";
    $("chartPoint").textContent = `${d.dates[i]} · รายวัน: ${value(
      d.usage[i]
    )} · เฉลี่ย 30 วัน: ${value(d.average[i])}`;
  };
  render();
}
