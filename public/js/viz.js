/**
 * Visualization library for MiniLLM
 * Canvas-based neuron rendering, connections, decision boundaries, training viz
 * @module Viz
 */
'use strict';

const Viz = {
  /** Color palette */
  colors: {
    purple: '#8b5cf6',
    blue: '#3b82f6',
    orange: '#f97316',
    green: '#22c55e',
    red: '#ef4444',
    pink: '#ec4899',
    bg: '#f8fafc',
    card: 'rgba(255,255,255,0.7)',
    text: '#1e293b',
    textLight: '#64748b',
    positive: '#3b82f6',
    negative: '#f97316'
  },

  /**
   * Draw a decision boundary heatmap from a grid
   * @param {CanvasRenderingContext2D} ctx
   * @param {Float32Array} grid - Flat array from classifyGrid
   * @param {number} resolution
   * @param {number} w - Canvas width
   * @param {number} h - Canvas height
   */
  drawDecisionBoundary(ctx, grid, resolution, w, h) {
    const cellW = w / resolution;
    const cellH = h / resolution;
    for (let i = 0; i < resolution; i++) {
      for (let j = 0; j < resolution; j++) {
        const val = grid[i * resolution + j];
        // Blue for class 1, orange for class 0
        const r = Math.round(59 + (249 - 59) * (1 - val));
        const g = Math.round(130 + (115 - 130) * (1 - val));
        const b = Math.round(246 + (22 - 246) * (1 - val));
        ctx.fillStyle = `rgba(${r},${g},${b},0.35)`;
        ctx.fillRect(j * cellW, i * cellH, cellW + 1, cellH + 1);
      }
    }
  },

  /**
   * Draw data points
   * @param {CanvasRenderingContext2D} ctx
   * @param {Array} data - Array of {input: [x,y], target: [label]}
   * @param {number} w - Canvas width
   * @param {number} h - Canvas height
   * @param {number[]} range - [xMin, xMax, yMin, yMax]
   */
  drawDataPoints(ctx, data, w, h, range = [-6, 6, -6, 6]) {
    const [xMin, xMax, yMin, yMax] = range;
    for (const pt of data) {
      const px = ((pt.input[0] - xMin) / (xMax - xMin)) * w;
      const py = ((pt.input[1] - yMin) / (yMax - yMin)) * h;
      const label = pt.target[0];

      ctx.beginPath();
      ctx.arc(px, py, 4, 0, Math.PI * 2);
      ctx.fillStyle = label > 0.5 ? Viz.colors.blue : Viz.colors.orange;
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  },

  /**
   * Draw a neuron (glowing orb)
   * @param {CanvasRenderingContext2D} ctx
   * @param {number} x
   * @param {number} y
   * @param {number} r - Radius
   * @param {number} activation - 0 to 1
   * @param {string} [color]
   */
  drawNeuron(ctx, x, y, r, activation = 0.5, color = null) {
    const baseColor = color || Viz.colors.purple;
    // Glow
    const glow = ctx.createRadialGradient(x, y, r * 0.3, x, y, r * 2.5);
    glow.addColorStop(0, Viz.hexToRgba(baseColor, activation * 0.4));
    glow.addColorStop(1, Viz.hexToRgba(baseColor, 0));
    ctx.fillStyle = glow;
    ctx.fillRect(x - r * 3, y - r * 3, r * 6, r * 6);

    // Body
    const grad = ctx.createRadialGradient(x - r * 0.3, y - r * 0.3, r * 0.1, x, y, r);
    grad.addColorStop(0, Viz.hexToRgba('#ffffff', 0.9));
    grad.addColorStop(0.5, Viz.hexToRgba(baseColor, 0.3 + activation * 0.5));
    grad.addColorStop(1, Viz.hexToRgba(baseColor, 0.5 + activation * 0.4));
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = grad;
    ctx.fill();

    // Border
    ctx.strokeStyle = Viz.hexToRgba(baseColor, 0.6);
    ctx.lineWidth = 2;
    ctx.stroke();
  },

  /**
   * Draw a connection line between neurons
   * @param {CanvasRenderingContext2D} ctx
   * @param {number} x1
   * @param {number} y1
   * @param {number} x2
   * @param {number} y2
   * @param {number} weight - Weight value
   * @param {number} [maxWeight=2]
   */
  drawConnection(ctx, x1, y1, x2, y2, weight, maxWeight = 2) {
    const norm = Math.min(Math.abs(weight) / maxWeight, 1);
    const color = weight >= 0 ? Viz.colors.blue : Viz.colors.orange;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.strokeStyle = Viz.hexToRgba(color, 0.2 + norm * 0.6);
    ctx.lineWidth = 0.5 + norm * 3;
    ctx.stroke();
  },

  /**
   * Draw a full network diagram
   * @param {CanvasRenderingContext2D} ctx
   * @param {import('./neural-engine').NeuralNetwork} net
   * @param {number} w
   * @param {number} h
   * @param {number[]} [inputVals] - Current input values for activation display
   */
  drawNetwork(ctx, net, w, h, inputVals = null) {
    ctx.clearRect(0, 0, w, h);
    const topology = net.topology;
    const numLayers = topology.length;
    const layerSpacing = w / (numLayers + 1);
    const maxNeurons = Math.max(...topology);
    const neuronR = Math.min(20, h / (maxNeurons * 3));

    // Compute positions
    const positions = [];
    for (let l = 0; l < numLayers; l++) {
      positions[l] = [];
      const n = topology[l];
      const totalH = n * neuronR * 3;
      const startY = (h - totalH) / 2 + neuronR * 1.5;
      for (let i = 0; i < n; i++) {
        positions[l][i] = {
          x: layerSpacing * (l + 1),
          y: startY + i * neuronR * 3
        };
      }
    }

    // Run forward pass if we have inputs
    let activations = [];
    if (inputVals) {
      activations[0] = inputVals;
      let out = inputVals;
      for (let l = 0; l < net.layers.length; l++) {
        out = net.layers[l].forward(out);
        activations[l + 1] = out;
      }
    }

    // Draw connections
    for (let l = 0; l < net.layers.length; l++) {
      const layer = net.layers[l];
      for (let i = 0; i < layer.outputSize; i++) {
        for (let j = 0; j < layer.inputSize; j++) {
          Viz.drawConnection(
            ctx,
            positions[l][j].x, positions[l][j].y,
            positions[l + 1][i].x, positions[l + 1][i].y,
            layer.weights[i][j]
          );
        }
      }
    }

    // Draw neurons
    for (let l = 0; l < numLayers; l++) {
      for (let i = 0; i < topology[l]; i++) {
        const act = activations[l] ? activations[l][i] : 0.5;
        const color = l === 0 ? Viz.colors.blue :
          l === numLayers - 1 ? Viz.colors.orange : Viz.colors.purple;
        Viz.drawNeuron(ctx, positions[l][i].x, positions[l][i].y, neuronR, Math.abs(act), color);

        // Label
        if (activations[l]) {
          ctx.fillStyle = Viz.colors.text;
          ctx.font = `${Math.max(9, neuronR * 0.6)}px Inter, sans-serif`;
          ctx.textAlign = 'center';
          ctx.fillText(act.toFixed(2), positions[l][i].x, positions[l][i].y + neuronR + 14);
        }
      }
    }
  },

  /**
   * Draw a loss chart
   * @param {CanvasRenderingContext2D} ctx
   * @param {number[]} history
   * @param {number} w
   * @param {number} h
   */
  drawLossChart(ctx, history, w, h) {
    if (!history.length) return;
    ctx.clearRect(0, 0, w, h);

    const maxLoss = Math.max(...history, 0.01);
    const padding = 40;
    const chartW = w - padding * 2;
    const chartH = h - padding * 2;

    // Grid
    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
      const y = padding + (chartH / 4) * i;
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(w - padding, y);
      ctx.stroke();
      ctx.fillStyle = Viz.colors.textLight;
      ctx.font = '10px Inter, sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText((maxLoss * (1 - i / 4)).toFixed(3), padding - 5, y + 3);
    }

    // Line
    ctx.beginPath();
    for (let i = 0; i < history.length; i++) {
      const x = padding + (i / Math.max(history.length - 1, 1)) * chartW;
      const y = padding + (1 - history[i] / maxLoss) * chartH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = Viz.colors.purple;
    ctx.lineWidth = 2;
    ctx.stroke();

    // Labels
    ctx.fillStyle = Viz.colors.text;
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Epoch', w / 2, h - 5);
    ctx.save();
    ctx.translate(12, h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Loss', 0, 0);
    ctx.restore();
  },

  /**
   * Convert hex to rgba string
   */
  hexToRgba(hex, alpha) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${alpha})`;
  }
};

if (typeof window !== 'undefined') {
  window.Viz = Viz;
}
