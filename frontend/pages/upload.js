// หน้านำเข้าข้อมูล: เลือกไฟล์ ตรวจสอบ และยืนยันนำเข้า
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

