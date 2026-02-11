// MiniLLM — TensorFlow.js Character-Level LSTM Trainer
(function() {
  'use strict';

  let trainingData = '';
  let model = null;
  let charToIdx = {};
  let idxToChar = {};
  let vocabSize = 0;
  let seqLength = 40;
  let isTraining = false;
  let lossHistory = [];

  // ===== Preset data loading =====
  const presets = {
    shakespeare: '/data/shakespeare.txt',
    recipes: '/data/recipes.txt',
    python: '/data/python.txt'
  };
  let activePreset = 'shakespeare';

  // Bind preset buttons
  document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const preset = btn.dataset.preset;
      activePreset = preset;

      const customArea = document.getElementById('custom-text');
      if (preset === 'custom') {
        customArea.style.display = 'block';
        trainingData = customArea.value;
        updatePreview();
      } else {
        customArea.style.display = 'none';
        loadPreset(preset);
      }
    });
  });

  const customArea = document.getElementById('custom-text');
  if (customArea) {
    customArea.addEventListener('input', () => {
      trainingData = customArea.value;
      updatePreview();
    });
  }

  async function loadPreset(name) {
    if (!presets[name]) return;
    try {
      const resp = await fetch(presets[name]);
      trainingData = await resp.text();
      updatePreview();
    } catch (e) {
      console.error('Failed to load preset:', e);
    }
  }

  function updatePreview() {
    const el = document.getElementById('preview-text');
    if (el) {
      el.textContent = trainingData.slice(0, 300) + (trainingData.length > 300 ? '...' : '');
    }
  }

  // Load default preset
  loadPreset('shakespeare');

  // Temperature slider
  const tempSlider = document.getElementById('temperature');
  const tempVal = document.getElementById('temp-val');
  if (tempSlider) {
    tempSlider.addEventListener('input', () => {
      if (tempVal) tempVal.textContent = parseFloat(tempSlider.value).toFixed(1);
    });
  }

  // ===== Build vocabulary =====
  function buildVocab(text) {
    const chars = [...new Set(text.split(''))].sort();
    charToIdx = {};
    idxToChar = {};
    chars.forEach((c, i) => { charToIdx[c] = i; idxToChar[i] = c; });
    vocabSize = chars.length;
    return chars;
  }

  // ===== Create model =====
  function createModel(size) {
    const units = size === 'tiny' ? 64 : size === 'small' ? 128 : 256;
    const numLayers = size === 'medium' ? 2 : 1;

    const mdl = tf.sequential();
    for (let i = 0; i < numLayers; i++) {
      mdl.add(tf.layers.lstm({
        units,
        returnSequences: i < numLayers - 1,
        inputShape: i === 0 ? [seqLength, vocabSize] : undefined
      }));
    }
    mdl.add(tf.layers.dense({ units: vocabSize, activation: 'softmax' }));
    mdl.compile({ optimizer: tf.train.adam(0.01), loss: 'categoricalCrossentropy' });
    return mdl;
  }

  // ===== Prepare training data =====
  function prepareData(text, batchSize) {
    const xs = [];
    const ys = [];
    const step = Math.max(3, Math.floor(text.length / (batchSize * seqLength)));

    for (let i = 0; i < text.length - seqLength - 1; i += step) {
      if (xs.length >= batchSize) break;
      const inputSeq = text.slice(i, i + seqLength);
      const target = text[i + seqLength];
      if (charToIdx[target] === undefined) continue;

      // One-hot encode input sequence
      const xArr = [];
      let valid = true;
      for (let j = 0; j < seqLength; j++) {
        const idx = charToIdx[inputSeq[j]];
        if (idx === undefined) { valid = false; break; }
        const oh = new Array(vocabSize).fill(0);
        oh[idx] = 1;
        xArr.push(oh);
      }
      if (!valid) continue;

      const yArr = new Array(vocabSize).fill(0);
      yArr[charToIdx[target]] = 1;

      xs.push(xArr);
      ys.push(yArr);
    }

    return {
      x: tf.tensor3d(xs),
      y: tf.tensor2d(ys)
    };
  }

  // ===== Generate text =====
  function generateText(prompt, length, temperature) {
    if (!model || vocabSize === 0) return '';

    temperature = temperature || 0.8;
    let input = prompt.slice(-seqLength).padStart(seqLength, ' ');
    let result = '';

    for (let i = 0; i < length; i++) {
      const xArr = [];
      for (let j = 0; j < seqLength; j++) {
        const oh = new Array(vocabSize).fill(0);
        const idx = charToIdx[input[j]];
        if (idx !== undefined) oh[idx] = 1;
        xArr.push(oh);
      }

      const pred = model.predict(tf.tensor3d([xArr]));
      const logits = pred.dataSync();

      // Temperature sampling
      const scaled = logits.map(l => Math.exp(Math.log(Math.max(l, 1e-8)) / temperature));
      const sum = scaled.reduce((a, b) => a + b, 0);
      const probs = scaled.map(s => s / sum);

      // Sample from distribution
      let r = Math.random();
      let idx = 0;
      for (let j = 0; j < probs.length; j++) {
        r -= probs[j];
        if (r <= 0) { idx = j; break; }
      }

      const char = idxToChar[idx] || ' ';
      result += char;
      input = input.slice(1) + char;

      pred.dispose();
    }

    return result;
  }

  // ===== Draw loss chart =====
  function drawLossChart() {
    const canvas = document.getElementById('loss-chart');
    if (!canvas || lossHistory.length === 0) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const maxLoss = Math.max(...lossHistory, 0.1);
    const minLoss = Math.min(...lossHistory);
    const range = maxLoss - minLoss || 1;

    // Fill gradient
    ctx.beginPath();
    ctx.moveTo(0, H);
    lossHistory.forEach((loss, i) => {
      const x = (i / (lossHistory.length - 1)) * W;
      const y = H - ((loss - minLoss) / range) * (H - 20) - 10;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.lineTo(W, H);
    ctx.closePath();
    const grad = ctx.createLinearGradient(0, 0, 0, H);
    grad.addColorStop(0, 'rgba(139, 92, 246, 0.2)');
    grad.addColorStop(1, 'rgba(139, 92, 246, 0)');
    ctx.fillStyle = grad;
    ctx.fill();

    // Line
    ctx.beginPath();
    lossHistory.forEach((loss, i) => {
      const x = (i / (lossHistory.length - 1)) * W;
      const y = H - ((loss - minLoss) / range) * (H - 20) - 10;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = '#8b5cf6';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Current value dot
    if (lossHistory.length > 0) {
      const lastLoss = lossHistory[lossHistory.length - 1];
      const x = W;
      const y = H - ((lastLoss - minLoss) / range) * (H - 20) - 10;
      ctx.fillStyle = '#8b5cf6';
      ctx.beginPath();
      ctx.arc(x - 2, y, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // ===== Training loop =====
  const trainBtn = document.getElementById('train-btn');
  if (trainBtn) {
    trainBtn.addEventListener('click', startTraining);
  }

  async function startTraining() {
    if (isTraining) return;
    if (trainingData.length < 500) {
      alert('Please provide at least 500 characters of training data.');
      return;
    }

    isTraining = true;
    trainBtn.disabled = true;
    trainBtn.textContent = '⏳ Training...';
    lossHistory = [];

    const dashboard = document.getElementById('training-dashboard');
    if (dashboard) dashboard.style.display = 'block';

    const sizeEl = document.getElementById('model-size');
    const size = sizeEl ? sizeEl.value : 'small';
    const epochs = size === 'tiny' ? 20 : size === 'small' ? 30 : 40;
    const batchSize = size === 'tiny' ? 64 : size === 'small' ? 128 : 128;

    // Build vocab and model
    buildVocab(trainingData);
    model = createModel(size);

    // Prepare data
    const data = prepareData(trainingData, batchSize);

    // Train
    for (let epoch = 0; epoch < epochs; epoch++) {
      if (!isTraining) break;

      const result = await model.fit(data.x, data.y, {
        epochs: 1,
        batchSize: 32,
        shuffle: true,
        verbose: 0
      });

      const loss = result.history.loss[0];
      lossHistory.push(loss);

      // Update UI
      const dashLoss = document.getElementById('dash-loss');
      const dashEpoch = document.getElementById('dash-epoch');
      const progressFill = document.getElementById('progress-fill');
      const sampleText = document.getElementById('sample-text');

      if (dashLoss) dashLoss.textContent = loss.toFixed(4);
      if (dashEpoch) dashEpoch.textContent = `${epoch + 1} / ${epochs}`;
      if (progressFill) progressFill.style.width = `${((epoch + 1) / epochs) * 100}%`;

      drawLossChart();

      // Generate sample every 5 epochs
      if ((epoch + 1) % 5 === 0 || epoch === epochs - 1) {
        const seed = trainingData.slice(0, seqLength);
        const sample = generateText(seed, 100, 0.8);
        if (sampleText) sampleText.textContent = seed + sample;
      }

      // Yield to UI
      await new Promise(r => setTimeout(r, 10));
    }

    // Cleanup tensors
    data.x.dispose();
    data.y.dispose();

    // Training complete — show generate UI
    trainBtn.textContent = '✅ Done!';
    isTraining = false;

    const step4 = document.getElementById('train-step-4');
    if (step4) step4.style.display = 'block';

    // Bind generate
    const genPrompt = document.getElementById('gen-prompt');
    const genOutput = document.getElementById('gen-output');
    if (genPrompt) {
      genPrompt.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          const prompt = genPrompt.value;
          const temp = tempSlider ? parseFloat(tempSlider.value) : 0.8;
          const output = generateText(prompt, 200, temp);
          if (genOutput) {
            genOutput.textContent = '';
            // Typewriter effect
            let i = 0;
            const type = () => {
              if (i < output.length) {
                genOutput.textContent += output[i];
                i++;
                setTimeout(type, 15);
              }
            };
            type();
          }
        }
      });
    }
  }

  // ===== Share model =====
  const shareBtn = document.getElementById('share-btn');
  if (shareBtn) {
    shareBtn.addEventListener('click', async () => {
      if (!model) return;
      try {
        // Save model to JSON
        const saveResult = await model.save(tf.io.withSaveHandler(async (artifacts) => {
          const modelData = {
            topology: artifacts.modelTopology,
            weightSpecs: artifacts.weightSpecs,
            weightData: Array.from(new Uint8Array(artifacts.weightData)),
            vocab: charToIdx,
            seqLength,
            preset: activePreset,
            lossHistory
          };

          const resp = await fetch('/api/models/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(modelData)
          });

          if (!resp.ok) throw new Error('Save failed');
          const result = await resp.json();

          const shareLink = document.getElementById('share-link');
          if (shareLink) {
            const url = `${window.location.origin}/model.html?id=${result.id}`;
            shareLink.innerHTML = `Share link: <a href="${url}" target="_blank">${url}</a>`;
            shareLink.style.display = 'block';
          }

          return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: 'JSON' } };
        }));
      } catch (e) {
        console.error('Share failed:', e);
        alert('Failed to share model. Try again.');
      }
    });
  }

  // ===== Download model =====
  const downloadBtn = document.getElementById('download-btn');
  if (downloadBtn) {
    downloadBtn.addEventListener('click', async () => {
      if (!model) return;
      await model.save('downloads://minillm-model');
    });
  }

})();
