// MiniLLM — Scroll-driven story + inline interactive neurons
(function() {
  'use strict';

  // ===== Scroll reveal =====
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) e.target.classList.add('visible');
    });
  }, { threshold: 0.15 });

  document.querySelectorAll('.reveal').forEach(el => observer.observe(el));

  // Show chapter rail after scrolling past hero
  const rail = document.getElementById('chapter-rail');
  const heroObs = new IntersectionObserver(([e]) => {
    rail.classList.toggle('visible', !e.isIntersecting);
  }, { threshold: 0.5 });
  heroObs.observe(document.getElementById('hero'));

  // ===== Sigmoid function =====
  function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

  // ===== Hero neuron (pulsing orb) =====
  const heroCanvas = document.getElementById('hero-neuron');
  if (heroCanvas) {
    const ctx = heroCanvas.getContext('2d');
    const W = heroCanvas.width, H = heroCanvas.height;
    let phase = 0;

    function drawHeroNeuron() {
      ctx.clearRect(0, 0, W, H);
      const cx = W / 2, cy = H / 2;
      phase += 0.02;
      const pulse = 1 + Math.sin(phase) * 0.08;
      const r = 60 * pulse;

      // Outer glow
      const grad = ctx.createRadialGradient(cx, cy, r * 0.3, cx, cy, r * 2.5);
      grad.addColorStop(0, 'rgba(139, 92, 246, 0.3)');
      grad.addColorStop(0.5, 'rgba(139, 92, 246, 0.08)');
      grad.addColorStop(1, 'rgba(139, 92, 246, 0)');
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(cx, cy, r * 2.5, 0, Math.PI * 2);
      ctx.fill();

      // Main orb
      const orbGrad = ctx.createRadialGradient(cx - r * 0.2, cy - r * 0.2, 0, cx, cy, r);
      orbGrad.addColorStop(0, '#c4b5fd');
      orbGrad.addColorStop(0.6, '#8b5cf6');
      orbGrad.addColorStop(1, '#6d28d9');
      ctx.fillStyle = orbGrad;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fill();

      // Inner highlight
      ctx.fillStyle = 'rgba(255,255,255,0.3)';
      ctx.beginPath();
      ctx.arc(cx - r * 0.25, cy - r * 0.25, r * 0.35, 0, Math.PI * 2);
      ctx.fill();

      // Floating particles
      for (let i = 0; i < 6; i++) {
        const angle = phase * 0.5 + (i * Math.PI * 2 / 6);
        const dist = r * 1.8 + Math.sin(phase + i) * 15;
        const px = cx + Math.cos(angle) * dist;
        const py = cy + Math.sin(angle) * dist;
        const pr = 2 + Math.sin(phase * 2 + i) * 1;
        ctx.fillStyle = `rgba(139, 92, 246, ${0.3 + Math.sin(phase + i) * 0.2})`;
        ctx.beginPath();
        ctx.arc(px, py, pr, 0, Math.PI * 2);
        ctx.fill();
      }

      requestAnimationFrame(drawHeroNeuron);
    }
    drawHeroNeuron();
  }

  // ===== Input neuron visualization =====
  const inputCanvas = document.getElementById('input-neuron');
  const weightCanvas = document.getElementById('weight-neuron');

  // Shared state
  const state = {
    input1: 3, input2: 7,
    weight1: 0.5, weight2: -0.3, bias: 0
  };

  function bindSlider(id, key) {
    const el = document.getElementById(id);
    const valEl = document.getElementById(id + '-val');
    if (!el) return;
    el.addEventListener('input', () => {
      state[key] = parseFloat(el.value);
      if (valEl) valEl.textContent = state[key].toFixed(1);
      updateVisualizations();
    });
  }
  bindSlider('input1', 'input1');
  bindSlider('input2', 'input2');
  bindSlider('weight1', 'weight1');
  bindSlider('weight2', 'weight2');
  bindSlider('bias', 'bias');

  function getOutput() {
    const sum = state.input1 * state.weight1 + state.input2 * state.weight2 + state.bias;
    return { sum, output: sigmoid(sum) };
  }

  function drawNeuronViz(canvas, showWeights) {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const { sum, output } = getOutput();
    const cx = W * 0.55, cy = H / 2;
    const r = 35;

    // Input lines
    const inputs = [
      { x: 50, y: H * 0.3, val: state.input1, w: state.weight1, label: `x₁ = ${state.input1.toFixed(1)}` },
      { x: 50, y: H * 0.7, val: state.input2, w: state.weight2, label: `x₂ = ${state.input2.toFixed(1)}` }
    ];

    inputs.forEach(inp => {
      // Connection line
      const alpha = showWeights ? Math.min(Math.abs(inp.w), 1) : 0.4;
      const color = inp.w >= 0 ? `rgba(59, 130, 246, ${alpha})` : `rgba(249, 115, 22, ${alpha})`;
      ctx.strokeStyle = color;
      ctx.lineWidth = showWeights ? Math.max(Math.abs(inp.w) * 4, 1) : 2;
      ctx.beginPath();
      ctx.moveTo(inp.x + 30, inp.y);
      ctx.lineTo(cx - r, cy);
      ctx.stroke();

      // Input circle
      ctx.fillStyle = 'rgba(59, 130, 246, 0.15)';
      ctx.beginPath();
      ctx.arc(inp.x + 15, inp.y, 15, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = 'rgba(59, 130, 246, 0.4)';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Input label
      ctx.fillStyle = '#3b82f6';
      ctx.font = '600 11px Inter';
      ctx.textAlign = 'center';
      ctx.fillText(inp.label, inp.x + 15, inp.y + 4);

      // Weight label (if showing weights)
      if (showWeights) {
        const mx = (inp.x + 30 + cx - r) / 2;
        const my = (inp.y + cy) / 2;
        ctx.fillStyle = inp.w >= 0 ? '#3b82f6' : '#f97316';
        ctx.font = 'bold 10px Inter';
        ctx.fillText(`w=${inp.w.toFixed(1)}`, mx, my - 8);
      }
    });

    // Main neuron orb
    const intensity = output;
    const orbGrad = ctx.createRadialGradient(cx - r * 0.2, cy - r * 0.2, 0, cx, cy, r);
    orbGrad.addColorStop(0, `rgba(196, 181, 253, ${0.5 + intensity * 0.5})`);
    orbGrad.addColorStop(0.6, `rgba(139, 92, 246, ${0.5 + intensity * 0.5})`);
    orbGrad.addColorStop(1, `rgba(109, 40, 217, ${0.4 + intensity * 0.6})`);

    // Glow
    const glowR = r * (1.5 + intensity * 0.5);
    const glow = ctx.createRadialGradient(cx, cy, r * 0.5, cx, cy, glowR);
    glow.addColorStop(0, `rgba(139, 92, 246, ${intensity * 0.25})`);
    glow.addColorStop(1, 'rgba(139, 92, 246, 0)');
    ctx.fillStyle = glow;
    ctx.beginPath();
    ctx.arc(cx, cy, glowR, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = orbGrad;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fill();

    // Highlight
    ctx.fillStyle = 'rgba(255,255,255,0.25)';
    ctx.beginPath();
    ctx.arc(cx - r * 0.2, cy - r * 0.2, r * 0.3, 0, Math.PI * 2);
    ctx.fill();

    // Output arrow
    ctx.strokeStyle = `rgba(16, 185, 129, ${0.3 + intensity * 0.7})`;
    ctx.lineWidth = 2 + intensity * 2;
    ctx.beginPath();
    ctx.moveTo(cx + r, cy);
    ctx.lineTo(W - 50, cy);
    ctx.stroke();

    // Output value
    ctx.fillStyle = '#10b981';
    ctx.font = 'bold 16px Inter';
    ctx.textAlign = 'center';
    ctx.fillText(output.toFixed(3), W - 30, cy + 5);
  }

  function drawSigmoidCurve() {
    const canvas = document.getElementById('sigmoid-curve');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const { sum, output } = getOutput();
    const pad = 20;

    // Axes
    ctx.strokeStyle = 'rgba(0,0,0,0.1)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad, H / 2);
    ctx.lineTo(W - pad, H / 2);
    ctx.moveTo(W / 2, pad);
    ctx.lineTo(W / 2, H - pad);
    ctx.stroke();

    // Curve
    ctx.strokeStyle = 'rgba(139, 92, 246, 0.5)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let px = pad; px <= W - pad; px++) {
      const x = ((px - pad) / (W - 2 * pad) - 0.5) * 12;
      const y = sigmoid(x);
      const py = H - pad - y * (H - 2 * pad);
      if (px === pad) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();

    // Current point
    const dotX = pad + ((sum / 12 + 0.5) * (W - 2 * pad));
    const dotY = H - pad - output * (H - 2 * pad);
    ctx.fillStyle = '#8b5cf6';
    ctx.beginPath();
    ctx.arc(Math.max(pad, Math.min(W - pad, dotX)), Math.max(pad, Math.min(H - pad, dotY)), 5, 0, Math.PI * 2);
    ctx.fill();
  }

  function updateVisualizations() {
    drawNeuronViz(inputCanvas, false);
    drawNeuronViz(weightCanvas, true);
    drawSigmoidCurve();

    const { sum, output } = getOutput();
    const sumEl = document.getElementById('weighted-sum');
    const outEl = document.getElementById('neuron-output');
    if (sumEl) sumEl.textContent = sum.toFixed(2);
    if (outEl) outEl.textContent = output.toFixed(3);
  }

  // Initial draw
  updateVisualizations();

  // ===== Network reveal (animated) =====
  const netCanvas = document.getElementById('network-reveal');
  if (netCanvas) {
    const ctx = netCanvas.getContext('2d');
    const W = netCanvas.width, H = netCanvas.height;
    const layers = [3, 6, 8, 6, 4, 2];
    let netPhase = 0;
    let netVisible = false;

    const netObs = new IntersectionObserver(([e]) => {
      netVisible = e.isIntersecting;
    }, { threshold: 0.2 });
    netObs.observe(netCanvas);

    function getNeuronPos(layer, neuron) {
      const x = 80 + (layer / (layers.length - 1)) * (W - 160);
      const count = layers[layer];
      const spacing = Math.min(50, (H - 80) / (count + 1));
      const startY = H / 2 - ((count - 1) * spacing) / 2;
      const y = startY + neuron * spacing;
      return { x, y };
    }

    function drawNetwork() {
      ctx.clearRect(0, 0, W, H);
      netPhase += 0.015;

      // Connections
      for (let l = 0; l < layers.length - 1; l++) {
        for (let i = 0; i < layers[l]; i++) {
          for (let j = 0; j < layers[l + 1]; j++) {
            const from = getNeuronPos(l, i);
            const to = getNeuronPos(l + 1, j);
            const wave = Math.sin(netPhase + l + i * 0.3 + j * 0.2);
            const alpha = 0.03 + wave * 0.03;
            ctx.strokeStyle = `rgba(139, 92, 246, ${alpha})`;
            ctx.lineWidth = 0.5 + wave * 0.3;
            ctx.beginPath();
            ctx.moveTo(from.x, from.y);
            ctx.lineTo(to.x, to.y);
            ctx.stroke();
          }
        }
      }

      // Neurons
      for (let l = 0; l < layers.length; l++) {
        for (let i = 0; i < layers[l]; i++) {
          const { x, y } = getNeuronPos(l, i);
          const wave = Math.sin(netPhase * 2 + l * 0.5 + i * 0.7);
          const r = 8 + wave * 2;

          // Glow
          const glow = ctx.createRadialGradient(x, y, r * 0.3, x, y, r * 2);
          glow.addColorStop(0, `rgba(139, 92, 246, ${0.15 + wave * 0.1})`);
          glow.addColorStop(1, 'rgba(139, 92, 246, 0)');
          ctx.fillStyle = glow;
          ctx.beginPath();
          ctx.arc(x, y, r * 2, 0, Math.PI * 2);
          ctx.fill();

          // Orb
          const orbGrad = ctx.createRadialGradient(x - 2, y - 2, 0, x, y, r);
          orbGrad.addColorStop(0, '#c4b5fd');
          orbGrad.addColorStop(1, '#7c3aed');
          ctx.fillStyle = orbGrad;
          ctx.beginPath();
          ctx.arc(x, y, r, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      // Data flowing (animated dots along connections)
      for (let l = 0; l < layers.length - 1; l++) {
        const from = getNeuronPos(l, Math.floor(layers[l] / 2));
        const to = getNeuronPos(l + 1, Math.floor(layers[l + 1] / 2));
        const t = ((netPhase * 0.5 + l * 0.3) % 1);
        const dx = from.x + (to.x - from.x) * t;
        const dy = from.y + (to.y - from.y) * t;
        ctx.fillStyle = `rgba(59, 130, 246, ${0.6 - t * 0.5})`;
        ctx.beginPath();
        ctx.arc(dx, dy, 3, 0, Math.PI * 2);
        ctx.fill();
      }

      if (netVisible) requestAnimationFrame(drawNetwork);
    }

    // Start when visible
    const startNet = new IntersectionObserver(([e]) => {
      if (e.isIntersecting) { drawNetwork(); startNet.disconnect(); }
    }, { threshold: 0.1 });
    startNet.observe(netCanvas);
  }

})();
