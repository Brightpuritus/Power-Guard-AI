// หน้ารายการตรวจสอบ: ตาราง กราฟ SHAP แบบฟอร์ม และคิวงาน
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
            featureDescription(r.feature)
          )}</small></span><div class="shap-track"><i class="${
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

// อัปเดตจำนวนงานในแต่ละสถานะจากสรุปชุดข้อมูล
function renderWorkSummary(s) {
  $("workTotal").textContent = num(s.work.total);
  $("workPending").textContent = num(s.work.pending);
  $("workProgress").textContent = num(s.work.in_progress);
  $("workDone").textContent = num(s.work.completed);
}
