/**
 * MiniLLM Neural Network Engine
 * Real forward pass, real backpropagation, real gradient descent.
 * Pure JavaScript — no dependencies.
 * @module NeuralEngine
 */

'use strict';

/**
 * Activation functions and their derivatives
 */
const Activations = {
  sigmoid: {
    fn: x => 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x)))),
    dfn: y => y * (1 - y), // takes output, not input
    name: 'Sigmoid'
  },
  relu: {
    fn: x => Math.max(0, x),
    dfn: y => y > 0 ? 1 : 0,
    name: 'ReLU'
  },
  tanh: {
    fn: x => Math.tanh(x),
    dfn: y => 1 - y * y,
    name: 'Tanh'
  },
  linear: {
    fn: x => x,
    dfn: () => 1,
    name: 'Linear'
  }
};

/**
 * Loss functions
 */
const Losses = {
  mse: {
    fn: (predicted, target) => {
      let sum = 0;
      for (let i = 0; i < predicted.length; i++) {
        const d = predicted[i] - target[i];
        sum += d * d;
      }
      return sum / predicted.length;
    },
    dfn: (predicted, target) => {
      return predicted.map((p, i) => 2 * (p - target[i]) / predicted.length);
    }
  },
  crossEntropy: {
    fn: (predicted, target) => {
      let sum = 0;
      for (let i = 0; i < predicted.length; i++) {
        const p = Math.max(1e-15, Math.min(1 - 1e-15, predicted[i]));
        sum += -(target[i] * Math.log(p) + (1 - target[i]) * Math.log(1 - p));
      }
      return sum / predicted.length;
    },
    dfn: (predicted, target) => {
      return predicted.map((p, i) => {
        const pc = Math.max(1e-15, Math.min(1 - 1e-15, p));
        return (-target[i] / pc + (1 - target[i]) / (1 - pc)) / predicted.length;
      });
    }
  }
};

/**
 * A single neuron layer
 */
class Layer {
  /**
   * @param {number} inputSize - Number of inputs
   * @param {number} outputSize - Number of neurons in this layer
   * @param {string} activation - Activation function name
   */
  constructor(inputSize, outputSize, activation = 'sigmoid') {
    this.inputSize = inputSize;
    this.outputSize = outputSize;
    this.activation = Activations[activation] || Activations.sigmoid;
    this.activationName = activation;

    // Xavier initialization
    const scale = Math.sqrt(2 / (inputSize + outputSize));
    this.weights = [];
    for (let i = 0; i < outputSize; i++) {
      this.weights[i] = [];
      for (let j = 0; j < inputSize; j++) {
        this.weights[i][j] = (Math.random() * 2 - 1) * scale;
      }
    }
    this.biases = new Array(outputSize).fill(0).map(() => (Math.random() * 2 - 1) * 0.1);

    // Gradients
    this.weightGrads = [];
    this.biasGrads = new Array(outputSize).fill(0);
    for (let i = 0; i < outputSize; i++) {
      this.weightGrads[i] = new Array(inputSize).fill(0);
    }

    // Cache for backprop
    this.input = null;
    this.output = null;
    this.preActivation = null;
  }

  /**
   * Forward pass
   * @param {number[]} input
   * @returns {number[]}
   */
  forward(input) {
    this.input = input.slice();
    this.preActivation = new Array(this.outputSize);
    this.output = new Array(this.outputSize);

    for (let i = 0; i < this.outputSize; i++) {
      let sum = this.biases[i];
      for (let j = 0; j < this.inputSize; j++) {
        sum += this.weights[i][j] * input[j];
      }
      this.preActivation[i] = sum;
      this.output[i] = this.activation.fn(sum);
    }
    return this.output;
  }

  /**
   * Backward pass — compute gradients and return error for previous layer
   * @param {number[]} outputError - dL/dOutput for this layer
   * @returns {number[]} dL/dInput for previous layer
   */
  backward(outputError) {
    const inputError = new Array(this.inputSize).fill(0);

    for (let i = 0; i < this.outputSize; i++) {
      // For sigmoid/tanh, dfn takes output; for relu, we use preActivation check
      let delta;
      if (this.activationName === 'relu') {
        delta = outputError[i] * (this.preActivation[i] > 0 ? 1 : 0);
      } else if (this.activationName === 'linear') {
        delta = outputError[i];
      } else {
        delta = outputError[i] * this.activation.dfn(this.output[i]);
      }

      this.biasGrads[i] += delta;
      for (let j = 0; j < this.inputSize; j++) {
        this.weightGrads[i][j] += delta * this.input[j];
        inputError[j] += delta * this.weights[i][j];
      }
    }

    return inputError;
  }

  /**
   * Apply accumulated gradients
   * @param {number} lr - Learning rate
   * @param {number} l2 - L2 regularization strength
   * @param {number} batchSize - Number of samples in batch
   */
  applyGradients(lr, l2 = 0, batchSize = 1) {
    for (let i = 0; i < this.outputSize; i++) {
      this.biases[i] -= lr * (this.biasGrads[i] / batchSize);
      this.biasGrads[i] = 0;
      for (let j = 0; j < this.inputSize; j++) {
        this.weights[i][j] -= lr * (this.weightGrads[i][j] / batchSize + l2 * this.weights[i][j]);
        this.weightGrads[i][j] = 0;
      }
    }
  }

  /** Reset gradients to zero */
  zeroGrad() {
    for (let i = 0; i < this.outputSize; i++) {
      this.biasGrads[i] = 0;
      for (let j = 0; j < this.inputSize; j++) {
        this.weightGrads[i][j] = 0;
      }
    }
  }
}

/**
 * Neural Network — a stack of layers
 */
class NeuralNetwork {
  /**
   * @param {number[]} topology - Array of layer sizes, e.g. [2, 4, 4, 1]
   * @param {string} activation - Default activation for hidden layers
   * @param {string} outputActivation - Activation for the output layer
   */
  constructor(topology, activation = 'sigmoid', outputActivation = 'sigmoid') {
    this.topology = topology.slice();
    this.layers = [];
    this.activationName = activation;
    this.outputActivationName = outputActivation;

    for (let i = 1; i < topology.length; i++) {
      const act = i === topology.length - 1 ? outputActivation : activation;
      this.layers.push(new Layer(topology[i - 1], topology[i], act));
    }

    this.epoch = 0;
    this.loss = Infinity;
  }

  /**
   * Forward pass through all layers
   * @param {number[]} input
   * @returns {number[]}
   */
  forward(input) {
    let out = input;
    for (const layer of this.layers) {
      out = layer.forward(out);
    }
    return out;
  }

  /**
   * Train on a batch of data for one epoch
   * @param {Array<{input: number[], target: number[]}>} data
   * @param {number} lr - Learning rate
   * @param {string} lossType - 'mse' or 'crossEntropy'
   * @param {number} l2 - L2 regularization
   * @returns {number} Average loss
   */
  trainBatch(data, lr = 0.1, lossType = 'mse', l2 = 0) {
    const lossFn = Losses[lossType] || Losses.mse;

    // Zero gradients
    for (const layer of this.layers) layer.zeroGrad();

    let totalLoss = 0;

    for (const sample of data) {
      const output = this.forward(sample.input);
      totalLoss += lossFn.fn(output, sample.target);

      // Backprop
      let error = lossFn.dfn(output, sample.target);
      for (let i = this.layers.length - 1; i >= 0; i--) {
        error = this.layers[i].backward(error);
      }
    }

    // Apply gradients
    for (const layer of this.layers) {
      layer.applyGradients(lr, l2, data.length);
    }

    this.loss = totalLoss / data.length;
    this.epoch++;
    return this.loss;
  }

  /**
   * Train for multiple epochs
   * @param {Array} data
   * @param {number} epochs
   * @param {number} lr
   * @param {string} lossType
   * @param {number} l2
   * @param {Function} [onEpoch] - Callback(epoch, loss)
   * @returns {number[]} Loss history
   */
  train(data, epochs, lr = 0.1, lossType = 'mse', l2 = 0, onEpoch = null) {
    const history = [];
    for (let e = 0; e < epochs; e++) {
      const loss = this.trainBatch(data, lr, lossType, l2);
      history.push(loss);
      if (onEpoch) onEpoch(this.epoch, loss);
    }
    return history;
  }

  /**
   * Get all weights as a flat structure for visualization
   * @returns {Object}
   */
  getWeights() {
    return this.layers.map((l, i) => ({
      layer: i,
      weights: l.weights.map(w => w.slice()),
      biases: l.biases.slice()
    }));
  }

  /**
   * Get total parameter count
   * @returns {number}
   */
  paramCount() {
    let count = 0;
    for (const layer of this.layers) {
      count += layer.outputSize * layer.inputSize + layer.outputSize;
    }
    return count;
  }

  /**
   * Classify a 2D grid for decision boundary visualization
   * @param {number} resolution - Grid size
   * @param {number} [xMin=-6] 
   * @param {number} [xMax=6]
   * @param {number} [yMin=-6]
   * @param {number} [yMax=6]
   * @returns {Float32Array} Flat array of output values
   */
  classifyGrid(resolution = 50, xMin = -6, xMax = 6, yMin = -6, yMax = 6) {
    const grid = new Float32Array(resolution * resolution);
    const xStep = (xMax - xMin) / resolution;
    const yStep = (yMax - yMin) / resolution;

    for (let i = 0; i < resolution; i++) {
      for (let j = 0; j < resolution; j++) {
        const x = xMin + j * xStep + xStep / 2;
        const y = yMin + i * yStep + yStep / 2;
        const out = this.forward([x, y]);
        grid[i * resolution + j] = out[0];
      }
    }
    return grid;
  }
}

/**
 * Dataset generators for the playground
 */
const Datasets = {
  circle: (n = 200) => {
    const data = [];
    for (let i = 0; i < n; i++) {
      const angle = Math.random() * Math.PI * 2;
      const isInner = Math.random() < 0.5;
      const r = isInner ? Math.random() * 2 : 2.5 + Math.random() * 2;
      const x = r * Math.cos(angle) + (Math.random() - 0.5) * 0.3;
      const y = r * Math.sin(angle) + (Math.random() - 0.5) * 0.3;
      data.push({ input: [x, y], target: [isInner ? 1 : 0] });
    }
    return data;
  },

  xor: (n = 200) => {
    const data = [];
    for (let i = 0; i < n; i++) {
      const x = Math.random() * 6 - 3;
      const y = Math.random() * 6 - 3;
      const label = (x > 0) !== (y > 0) ? 1 : 0;
      data.push({ input: [x + (Math.random() - 0.5) * 0.5, y + (Math.random() - 0.5) * 0.5], target: [label] });
    }
    return data;
  },

  spiral: (n = 200) => {
    const data = [];
    const half = Math.floor(n / 2);
    for (let c = 0; c < 2; c++) {
      for (let i = 0; i < half; i++) {
        const r = (i / half) * 5;
        const t = 1.75 * i / half * 2 * Math.PI + (c * Math.PI);
        const x = r * Math.sin(t) + (Math.random() - 0.5) * 0.5;
        const y = r * Math.cos(t) + (Math.random() - 0.5) * 0.5;
        data.push({ input: [x, y], target: [c] });
      }
    }
    return data;
  },

  gaussian: (n = 200) => {
    const data = [];
    const half = Math.floor(n / 2);
    const randn = () => {
      let u = 0, v = 0;
      while (u === 0) u = Math.random();
      while (v === 0) v = Math.random();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    };
    for (let i = 0; i < half; i++) {
      data.push({ input: [randn() * 1.2 + 2, randn() * 1.2 + 2], target: [1] });
      data.push({ input: [randn() * 1.2 - 2, randn() * 1.2 - 2], target: [0] });
    }
    return data;
  },

  checkerboard: (n = 200) => {
    const data = [];
    for (let i = 0; i < n; i++) {
      const x = Math.random() * 8 - 4;
      const y = Math.random() * 8 - 4;
      const cx = Math.floor(x + 4);
      const cy = Math.floor(y + 4);
      const label = (cx + cy) % 2;
      data.push({ input: [x, y], target: [label] });
    }
    return data;
  }
};

// Single neuron helper for chapter 1
class SingleNeuron {
  constructor(numInputs = 1, activation = 'sigmoid') {
    this.weights = Array.from({ length: numInputs }, () => Math.random() * 2 - 1);
    this.bias = 0;
    this.activation = Activations[activation] || Activations.sigmoid;
  }

  forward(inputs) {
    let sum = this.bias;
    for (let i = 0; i < inputs.length; i++) {
      sum += this.weights[i] * inputs[i];
    }
    this.lastSum = sum;
    this.lastOutput = this.activation.fn(sum);
    return this.lastOutput;
  }
}

// Export for both module and browser
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { NeuralNetwork, Layer, SingleNeuron, Activations, Losses, Datasets };
}
if (typeof window !== 'undefined') {
  window.NeuralEngine = { NeuralNetwork, Layer, SingleNeuron, Activations, Losses, Datasets };
}
