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
    renderWorkSummary(s);
    renderOverview(s);
    renderModelInfo(s);
  } catch (e) {
    error(e);
  } finally {
    $("loading").hidden = true;
  }
}
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
// เริ่มแอปหลังสคริปต์ย่อยทั้งสี่ไฟล์ผูกเหตุการณ์เสร็จแล้ว
window.addEventListener("DOMContentLoaded", () => {
  view("upload");
  $("loading").hidden = true;
  loadDatasets().catch(error);
});
