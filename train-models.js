#!/usr/bin/env node
// Pre-train small LSTM language models for each preset
// Run once: node train-models.js
// Saves models to /opt/minillm/models/<preset>/

const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

const PRESETS = ['shakespeare', 'recipes', 'python'];
const SEQ_LEN = 30;
const UNITS = 64;
const EPOCHS = 30;
const BATCH_SIZE = 64;

async function trainModel(preset) {
  console.log(`\n=== Training: ${preset} ===`);
  const text = fs.readFileSync(path.join(__dirname, 'public', 'data', `${preset}.txt`), 'utf-8');
  console.log(`Text length: ${text.length} chars`);

  // Build vocabulary
  const chars = [...new Set(text.split(''))].sort();
  const charToIdx = {};
  const idxToChar = {};
  chars.forEach((c, i) => { charToIdx[c] = i; idxToChar[i] = c; });
  const vocabSize = chars.length;
  console.log(`Vocabulary: ${vocabSize} characters`);

  // Prepare training data (sample to keep memory low)
  const maxSamples = 2000;
  const step = Math.max(1, Math.floor((text.length - SEQ_LEN - 1) / maxSamples));
  const xData = [];
  const yData = [];

  for (let i = 0; i < text.length - SEQ_LEN - 1; i += step) {
    if (xData.length >= maxSamples) break;
    const seq = text.slice(i, i + SEQ_LEN);
    const target = text[i + SEQ_LEN];
    if (charToIdx[target] === undefined) continue;

    // One-hot encode
    const x = [];
    let valid = true;
    for (let j = 0; j < SEQ_LEN; j++) {
      if (charToIdx[seq[j]] === undefined) { valid = false; break; }
      const oh = new Float32Array(vocabSize);
      oh[charToIdx[seq[j]]] = 1;
      x.push(oh);
    }
    if (!valid) continue;

    const y = new Float32Array(vocabSize);
    y[charToIdx[target]] = 1;

    xData.push(x);
    yData.push(y);
  }

  console.log(`Training samples: ${xData.length}`);

  // Create tensors
  const xTensor = tf.tensor3d(xData);
  const yTensor = tf.tensor2d(yData);

  // Build model
  const model = tf.sequential();
  model.add(tf.layers.lstm({
    units: UNITS,
    inputShape: [SEQ_LEN, vocabSize],
    returnSequences: false
  }));
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.add(tf.layers.dense({ units: vocabSize, activation: 'softmax' }));

  model.compile({
    optimizer: tf.train.adam(0.005),
    loss: 'categoricalCrossentropy'
  });

  model.summary();

  // Train
  await model.fit(xTensor, yTensor, {
    epochs: EPOCHS,
    batchSize: BATCH_SIZE,
    shuffle: true,
    verbose: 1,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        if ((epoch + 1) % 5 === 0) {
          console.log(`  Epoch ${epoch + 1}/${EPOCHS} — loss: ${logs.loss.toFixed(4)}`);
        }
      }
    }
  });

  // Save model + vocab
  const modelDir = path.join(__dirname, 'models', preset);
  fs.mkdirSync(modelDir, { recursive: true });

  await model.save(`file://${modelDir}`);
  fs.writeFileSync(path.join(modelDir, 'vocab.json'), JSON.stringify({
    charToIdx, idxToChar, vocabSize, seqLen: SEQ_LEN
  }));

  console.log(`Model saved to ${modelDir}`);

  // Test generation
  let ctx = text.slice(0, SEQ_LEN);
  let gen = '';
  for (let i = 0; i < 100; i++) {
    const x = [];
    for (let j = 0; j < SEQ_LEN; j++) {
      const oh = new Float32Array(vocabSize);
      const idx = charToIdx[ctx[j]];
      if (idx !== undefined) oh[idx] = 1;
      x.push(oh);
    }
    const pred = model.predict(tf.tensor3d([x]));
    const probs = pred.dataSync();

    // Temperature sampling (0.7)
    const temp = 0.7;
    const scaled = Array.from(probs).map(p => Math.exp(Math.log(Math.max(p, 1e-8)) / temp));
    const sum = scaled.reduce((a, b) => a + b, 0);
    const normalized = scaled.map(s => s / sum);
    let r = Math.random();
    let idx = 0;
    for (let j = 0; j < normalized.length; j++) {
      r -= normalized[j];
      if (r <= 0) { idx = j; break; }
    }
    const ch = idxToChar[idx] || ' ';
    gen += ch;
    ctx = ctx.slice(1) + ch;
    pred.dispose();
  }
  console.log(`Sample: "${text.slice(0, SEQ_LEN)}${gen}"`);

  // Cleanup
  xTensor.dispose();
  yTensor.dispose();
  model.dispose();
}

(async () => {
  for (const preset of PRESETS) {
    await trainModel(preset);
  }
  console.log('\nAll models trained!');
})();
