/**
 * ICU-LNN 死亡风险预测 — 前端逻辑
 */

// ========== 全局状态 ==========
let gaugeChart = null;

// ========== 初始化 ==========
document.addEventListener("DOMContentLoaded", () => {
  initGauge();
  bindEvents();
});

// ========== 风险仪表盘 ==========
function initGauge() {
  const ctx = document.getElementById("risk-gauge").getContext("2d");
  gaugeChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      datasets: [{
        data: [0, 100],
        backgroundColor: ["#ef4444", "#e2e8f0"],
        borderWidth: 0,
        circumference: 270,
        rotation: 225,
      }]
    },
    options: {
      cutout: "75%",
      responsive: true,
      maintainAspectRatio: true,
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
    }
  });
}

function updateGauge(probability) {
  const p = Math.min(Math.max(probability, 0), 100);
  let color;
  if (p < 15) color = "#22c55e";
  else if (p < 35) color = "#eab308";
  else if (p < 60) color = "#f97316";
  else color = "#ef4444";

  gaugeChart.data.datasets[0].data = [p, 100 - p];
  gaugeChart.data.datasets[0].backgroundColor = [color, "#e2e8f0"];
  gaugeChart.update();
}

// ========== DOM元素 ==========
function bindEvents() {
  document.getElementById("predict-form").addEventListener("submit", handleSubmit);
  document.getElementById("btn-reset").addEventListener("click", resetForm);
  document.getElementById("btn-demo-high").addEventListener("click", loadHighRiskDemo);
  document.getElementById("btn-demo-low").addEventListener("click", loadLowRiskDemo);
}

// ========== 表单数据收集 ==========
function collectFormData() {
  return {
    age: document.getElementById("age").value,
    gender: document.getElementById("gender").value,
    bmi: document.getElementById("bmi").value,
    emergency: document.getElementById("emergency").value,
    charlson: document.getElementById("charlson").value,
    sofa: document.getElementById("sofa").value,
    sbp: document.getElementById("sbp").value,
    hr: document.getElementById("hr").value,
    temp: document.getElementById("temp").value,
    vital_hr: document.getElementById("vital_hr").value,
    vital_sbp: document.getElementById("vital_sbp").value,
    vital_dbp: document.getElementById("vital_dbp").value,
    vital_rr: document.getElementById("vital_rr").value,
    vital_spo2: document.getElementById("vital_spo2").value,
    vital_temp: document.getElementById("vital_temp").value,
    vital_gcs: document.getElementById("vital_gcs").value,
    vital_urine: document.getElementById("vital_urine").value,
    vital_lactate: document.getElementById("vital_lactate").value,
    vital_ph: document.getElementById("vital_ph").value,
  };
}

// ========== 预测处理 ==========
async function handleSubmit(e) {
  e.preventDefault();

  // 显示加载
  showState("loading");

  const formData = collectFormData();

  try {
    // 并行调用两个模型
    const [lrRes, cfcRes] = await Promise.all([
      fetch("/api/predict/lr", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      }).then(r => r.json()),
      // CfC需要3个时间点数据，使用相同数据近似
      fetch("/api/predict/cfc", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildCfcPayload(formData)),
      }).then(r => r.json()),
    ]);

    if (!lrRes.success) {
      throw new Error(lrRes.error || "预测失败");
    }

    renderResult(lrRes, cfcRes);
    showState("result");

  } catch (err) {
    showState("error");
    document.getElementById("error-message").textContent = err.message || "未知错误";
  }
}

function buildCfcPayload(formData) {
  // 将单时间点数据扩展为3个时间点序列（模拟趋势）
  const payload = { ...formData };
  for (const key of ["vital_hr", "vital_sbp", "vital_dbp", "vital_rr",
                      "vital_spo2", "vital_temp", "vital_gcs", "vital_urine",
                      "vital_lactate", "vital_ph"]) {
    payload[key + "_t0"] = formData[key];
    payload[key + "_t1"] = formData[key];
    payload[key + "_t2"] = formData[key];
  }
  return payload;
}

// ========== 结果渲染 ==========
function renderResult(lrRes, cfcRes) {
  const prob = lrRes.probability;
  const risk = lrRes.risk;

  // 更新仪表盘
  updateGauge(prob);

  // 百分比
  const pctEl = document.getElementById("risk-percentage");
  pctEl.textContent = prob.toFixed(1) + "%";
  pctEl.style.color = risk.color;

  // 风险等级标签
  const labelEl = document.getElementById("risk-label");
  labelEl.textContent = risk.icon + " " + risk.label;
  labelEl.style.color = risk.color;

  // 风险描述卡片
  const descCard = document.getElementById("risk-description-card");
  descCard.className = "detail-card risk-" + risk.level;
  document.getElementById("risk-description").textContent = risk.description;

  // 临床解读
  const ul = document.getElementById("interpretation-list");
  ul.innerHTML = "";
  if (lrRes.interpretation) {
    lrRes.interpretation.forEach(item => {
      const li = document.createElement("li");
      li.textContent = item;
      ul.appendChild(li);
    });
  }

  // 模型对比
  document.getElementById("lr-prob").textContent = prob.toFixed(1) + "%";
  document.getElementById("lr-prob").style.color = risk.color;
  document.getElementById("lr-card").style.borderColor = risk.color;

  if (cfcRes.success) {
    document.getElementById("cfc-prob").textContent = cfcRes.probability.toFixed(1) + "%";
    const cfcRisk = cfcRes.risk;
    document.getElementById("cfc-prob").style.color = cfcRisk ? cfcRisk.color : "#64748b";
  } else {
    document.getElementById("cfc-prob").textContent = "N/A";
  }

  // 模型名称
  document.getElementById("model-name").textContent = "LogisticRegression + CfC LNN";

  // 建议操作
  const actionBox = document.getElementById("action-box");
  const actionText = document.getElementById("action-text");

  if (prob < 15) {
    actionBox.style.background = "#f0fdf4";
    actionText.textContent = "建议：继续常规监护。患者生命体征稳定，可在标准流程下管理。每4小时复评一次。";
  } else if (prob < 35) {
    actionBox.style.background = "#fffbeb";
    actionText.textContent = "建议：加强监护频率至每小时1次，通知主管医师关注。评估是否需要调整治疗方案。";
  } else if (prob < 60) {
    actionBox.style.background = "#fff7ed";
    actionText.textContent = "⚠️ 建议：立即启动ICU预警，通知主治医师团队。进行快速序贯器官衰竭评估(qSOFA)，准备可能的升级治疗。";
  } else {
    actionBox.style.background = "#fef2f2";
    actionText.textContent = "🚨 建议：立即启动急救响应！通知ICU主任及急救团队，全面评估器官功能状态。准备机械通气、血管活性药物等高级生命支持。";
  }
}

// ========== 状态切换 ==========
function showState(state) {
  document.getElementById("empty-state").style.display = state === "empty" ? "flex" : "none";
  document.getElementById("loading-state").style.display = state === "loading" ? "flex" : "none";
  document.getElementById("error-state").style.display = state === "error" ? "flex" : "none";
  document.getElementById("prediction-result").style.display = state === "result" ? "block" : "none";
  document.getElementById("btn-predict").disabled = state === "loading";
}

// ========== 重置 ==========
function resetForm() {
  document.getElementById("predict-form").reset();
  document.getElementById("age").value = 65;
  document.getElementById("bmi").value = 28;
  document.getElementById("charlson").value = 3;
  document.getElementById("sofa").value = 5;
  document.getElementById("sbp").value = 120;
  document.getElementById("hr").value = 80;
  document.getElementById("temp").value = 37.0;
  document.getElementById("vital_hr").value = 80;
  document.getElementById("vital_sbp").value = 120;
  document.getElementById("vital_dbp").value = 70;
  document.getElementById("vital_rr").value = 18;
  document.getElementById("vital_spo2").value = 97;
  document.getElementById("vital_temp").value = 37.0;
  document.getElementById("vital_gcs").value = 15;
  document.getElementById("vital_urine").value = 50;
  document.getElementById("vital_lactate").value = 1.5;
  document.getElementById("vital_ph").value = 7.40;
  showState("empty");
  document.getElementById("model-name").textContent = "等待预测...";
}

// ========== 高危示例 ==========
function loadHighRiskDemo() {
  document.getElementById("age").value = 78;
  document.getElementById("bmi").value = 24;
  document.getElementById("emergency").value = "1";
  document.getElementById("charlson").value = 6;
  document.getElementById("sofa").value = 10;
  document.getElementById("sbp").value = 85;
  document.getElementById("hr").value = 115;
  document.getElementById("temp").value = 38.5;
  document.getElementById("vital_hr").value = 118;
  document.getElementById("vital_sbp").value = 78;
  document.getElementById("vital_dbp").value = 48;
  document.getElementById("vital_rr").value = 28;
  document.getElementById("vital_spo2").value = 88;
  document.getElementById("vital_temp").value = 38.8;
  document.getElementById("vital_gcs").value = 9;
  document.getElementById("vital_urine").value = 15;
  document.getElementById("vital_lactate").value = 4.5;
  document.getElementById("vital_ph").value = 7.18;
  // 自动提交
  document.getElementById("predict-form").dispatchEvent(new Event("submit"));
}

// ========== 低危示例 ==========
function loadLowRiskDemo() {
  document.getElementById("age").value = 45;
  document.getElementById("bmi").value = 26;
  document.getElementById("emergency").value = "0";
  document.getElementById("charlson").value = 0;
  document.getElementById("sofa").value = 1;
  document.getElementById("sbp").value = 125;
  document.getElementById("hr").value = 72;
  document.getElementById("temp").value = 36.8;
  document.getElementById("vital_hr").value = 74;
  document.getElementById("vital_sbp").value = 122;
  document.getElementById("vital_dbp").value = 76;
  document.getElementById("vital_rr").value = 16;
  document.getElementById("vital_spo2").value = 99;
  document.getElementById("vital_temp").value = 36.9;
  document.getElementById("vital_gcs").value = 15;
  document.getElementById("vital_urine").value = 65;
  document.getElementById("vital_lactate").value = 0.9;
  document.getElementById("vital_ph").value = 7.41;
  // 自动提交
  document.getElementById("predict-form").dispatchEvent(new Event("submit"));
}
