const tabs = document.querySelectorAll(".tab");
const panels = document.querySelectorAll(".panel");

const setActiveTab = (targetId) => {
  tabs.forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.tab === targetId);
  });
  panels.forEach((panel) => {
    panel.classList.toggle("active", panel.id === targetId);
  });
};

tabs.forEach((tab) => {
  tab.addEventListener("click", () => setActiveTab(tab.dataset.tab));
});

const formatNumber = (value, digits = 2) => Number(value).toFixed(digits);

const drawSparkline = (container, values) => {
  if (!container) return;
  container.innerHTML = "";
  const canvas = document.createElement("canvas");
  container.appendChild(canvas);
  const ctx = canvas.getContext("2d");
  const { width, height } = container.getBoundingClientRect();
  canvas.width = width;
  canvas.height = height;

  const max = Math.max(...values);
  const min = Math.min(...values);
  const range = max - min || 1;

  ctx.lineWidth = 3;
  ctx.strokeStyle = "#4b5ac7";
  ctx.beginPath();
  values.forEach((value, index) => {
    const x = (index / (values.length - 1)) * width;
    const y = height - ((value - min) / range) * height;
    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.stroke();
};

const updateKinetic = () => {
  const c0 = Number(document.querySelector('[data-input="kinetic-c0"]').value);
  const k = Number(document.querySelector('[data-input="kinetic-k"]').value);
  const t = Number(document.querySelector('[data-input="kinetic-t"]').value);
  const result = c0 * Math.exp(-k * t);
  document.querySelector('[data-output="kinetic-result"]').textContent = `${formatNumber(
    result,
    2
  )} mg/L`;
  const values = Array.from({ length: 20 }, (_, i) => c0 * Math.exp(-k * (i * (t / 19))));
  drawSparkline(document.querySelector('[data-chart="kinetic-chart"]'), values);
};

const updateArrhenius = () => {
  const ea = Number(document.querySelector('[data-input="arrhenius-ea"]').value);
  const temp = Number(document.querySelector('[data-input="arrhenius-temp"]').value);
  const r = 8.314;
  const tempK = temp + 273.15;
  const k0 = 1.2e7;
  const rate = k0 * Math.exp((-ea * 1000) / (r * tempK));
  document.querySelector('[data-output="arrhenius-result"]').textContent = `${formatNumber(rate, 4)} 1/h`;
  document.querySelector(
    '[data-output="arrhenius-note"]'
  ).textContent = `At ${temp}°C, higher activation energy lowers the rate constant.`;
};

const updateTwoStage = () => {
  const k1 = Number(document.querySelector('[data-input="twostage-k1"]').value);
  const k2 = Number(document.querySelector('[data-input="twostage-k2"]').value);
  const ts = Number(document.querySelector('[data-input="twostage-ts"]').value);
  const c0 = 100;
  const times = Array.from({ length: 30 }, (_, i) => i * 0.5);
  const values = times.map((time) => {
    if (time <= ts) {
      return c0 * Math.exp(-k1 * time);
    }
    const cSwitch = c0 * Math.exp(-k1 * ts);
    return cSwitch * Math.exp(-k2 * (time - ts));
  });
  const result = values[values.length - 1];
  document.querySelector('[data-output="twostage-result"]').textContent = `${formatNumber(
    result,
    2
  )} units`;
  drawSparkline(document.querySelector('[data-chart="twostage-chart"]'), values);
};

const updateMicrobial = () => {
  const max = Number(document.querySelector('[data-input="micro-max"]').value);
  const rate = Number(document.querySelector('[data-input="micro-rate"]').value);
  const lag = Number(document.querySelector('[data-input="micro-lag"]').value);
  const t = 12;
  const growth = max / (1 + Math.exp(-rate * (t - lag)));
  document.querySelector('[data-output="micro-result"]').textContent = `${formatNumber(
    growth,
    2
  )} log CFU`;
  const values = Array.from({ length: 24 }, (_, i) => {
    const time = i * 0.5;
    return max / (1 + Math.exp(-rate * (time - lag)));
  });
  drawSparkline(document.querySelector('[data-chart="micro-chart"]'), values);
};

const updateEnzyme = () => {
  const vmax = Number(document.querySelector('[data-input="enzyme-vmax"]').value);
  const km = Number(document.querySelector('[data-input="enzyme-km"]').value);
  const s = Number(document.querySelector('[data-input="enzyme-s"]').value);
  const velocity = (vmax * s) / (km + s);
  document.querySelector('[data-output="enzyme-result"]').textContent = `${formatNumber(
    velocity,
    2
  )} units`;
  const values = Array.from({ length: 20 }, (_, i) => (vmax * (i + 1)) / (km + i + 1));
  drawSparkline(document.querySelector('[data-chart="enzyme-chart"]'), values);
};

const updateGradient = () => {
  const rate = Number(document.querySelector('[data-input="gd-rate"]').value);
  const iter = Number(document.querySelector('[data-input="gd-iter"]').value);
  let loss = 8;
  const values = [];
  for (let i = 0; i < iter; i += 1) {
    loss *= 1 - rate * 0.08;
    values.push(loss);
  }
  document.querySelector('[data-output="gd-result"]').textContent = `${formatNumber(
    loss,
    3
  )}`;
  drawSparkline(document.querySelector('[data-chart="gd-chart"]'), values);
};

const updateRidge = () => {
  const lambda = Number(document.querySelector('[data-input="ridge-lambda"]').value);
  const noise = Number(document.querySelector('[data-input="ridge-noise"]').value);
  const score = Math.max(0, 1 - 0.08 * lambda - 0.06 * noise);
  document.querySelector('[data-output="ridge-result"]').textContent = `${formatNumber(
    score * 100,
    1
  )}%`;
  document.querySelector('[data-output="ridge-note"]').textContent =
    score > 0.7 ? "Strong generalization balance." : "Consider adjusting λ for stability.";
};

const updatePca = () => {
  const components = Number(document.querySelector('[data-input="pca-components"]').value);
  const signal = Number(document.querySelector('[data-input="pca-signal"]').value);
  const values = Array.from({ length: 6 }, (_, i) => Math.max(0, signal - i * 0.12));
  const total = values.reduce((sum, val) => sum + val, 0) || 1;
  const explained = values.slice(0, components).reduce((sum, val) => sum + val, 0) / total;
  document.querySelector('[data-output="pca-result"]').textContent = `${formatNumber(
    explained * 100,
    1
  )}%`;
  drawSparkline(document.querySelector('[data-chart="pca-chart"]'), values);
};

const updates = [
  updateKinetic,
  updateArrhenius,
  updateTwoStage,
  updateMicrobial,
  updateEnzyme,
  updateGradient,
  updateRidge,
  updatePca,
];

document.querySelectorAll("input[type='range']").forEach((input) => {
  input.addEventListener("input", () => {
    updates.forEach((update) => update());
  });
});

updates.forEach((update) => update());
