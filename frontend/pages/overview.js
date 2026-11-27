// หน้าศูนย์ปฏิบัติงาน: ใช้ helpers และสถานะกลางจาก app.js
// แสดงจำนวนและสัดส่วนความเสี่ยง รวมจำนวนรายที่ข้าม
function renderOverview(s) {
  $("scopeNote").textContent =
    "แสดงผลคัดกรองของชุดข้อมูลที่เลือก จำนวนผู้มีความเสี่ยงไม่ใช่จำนวนผู้ยืนยันว่าลักลอบใช้ไฟ";
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
}
