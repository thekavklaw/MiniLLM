const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

const SEQ_LEN = 30, UNITS = 64, EPOCHS = 30, BATCH_SIZE = 32;

(async () => {
  const text = fs.readFileSync(path.join(__dirname, 'public', 'data', 'python.txt'), 'utf-8');
  const chars = [...new Set(text.split(''))].sort();
  const charToIdx = {}, idxToChar = {};
  chars.forEach((c, i) => { charToIdx[c] = i; idxToChar[i] = c; });
  const vocabSize = chars.length;
  console.log(`Vocab: ${vocabSize}, Text: ${text.length}`);

  const maxSamples = 500; // Reduced to avoid OOM
  const step = Math.max(1, Math.floor((text.length - SEQ_LEN - 1) / maxSamples));
  const xData = [], yData = [];
  for (let i = 0; i < text.length - SEQ_LEN - 1; i += step) {
    if (xData.length >= maxSamples) break;
    const seq = text.slice(i, i + SEQ_LEN);
    const target = text[i + SEQ_LEN];
    if (charToIdx[target] === undefined) continue;
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
    xData.push(x); yData.push(y);
  }
  console.log(`Samples: ${xData.length}`);

  const xTensor = tf.tensor3d(xData);
  const yTensor = tf.tensor2d(yData);

  const model = tf.sequential();
  model.add(tf.layers.lstm({ units: UNITS, inputShape: [SEQ_LEN, vocabSize] }));
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.add(tf.layers.dense({ units: vocabSize, activation: 'softmax' }));
  model.compile({ optimizer: tf.train.adam(0.005), loss: 'categoricalCrossentropy' });

  await model.fit(xTensor, yTensor, { epochs: EPOCHS, batchSize: BATCH_SIZE, shuffle: true, verbose: 1 });

  const modelDir = path.join(__dirname, 'models', 'python');
  fs.mkdirSync(modelDir, { recursive: true });
  await model.save(`file://${modelDir}`);
  fs.writeFileSync(path.join(modelDir, 'vocab.json'), JSON.stringify({ charToIdx, idxToChar, vocabSize, seqLen: SEQ_LEN }));
  console.log('Python model saved!');

  xTensor.dispose(); yTensor.dispose(); model.dispose();
})();
