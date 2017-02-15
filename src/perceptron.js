export default class Perceptron {
  constructor(options = {}) {
    this.data = null;
    this.sizes = null;
    Object.assign(this, Perceptron.defaults, options);
  }

  train(data) {
    const inputSize = data[0].input.length;
    const outputSize = data[0].output.length;
    this.sizes = [inputSize, outputSize];
    this.data = data.slice(0);
    let success = true;
    for(let i = 0; i < this.data.length; i++) {
      const training = this.data.shift();
      success = this.trainPattern(training.input, training.output) && success;
      if (i === this.data.length - 1 && !success) {
        i = -1;
        console.log('restarting');
      }
    }
    return success;
  }

  trainPattern(input, output) {
    const { weights } = this;
    while (weights.length < input.length) {
      weights.push(Math.random());
    }

    // add a bias weight for the threshold
    if (weights.length === input.length) {
      weights.push(this.bias);
    }

    const result = this.run(input);
    this.data.push({ input, output, prev: result});

    if (result === output) return true;

    for (let i = 0; i < weights.length; i++) {
      const deltaInput = (i === input.length)
        ? this.threshold
        : input[i];
      weights[i] = this.delta(result, output, deltaInput, i);
    }
    return false
  }

  delta(actual, expected, input, learningRate) {
    return (expected - actual) * learningRate * input;
  }

  run(inputs) {
    let result = 0;
    for(let i = 0; i < inputs.length; i++) {
      result += inputs[i] * this.weights[i];
    }
    result += this.threshold * this.weights[this.weights.length - 1];
    return result > 0 ? 1 : 0;
  }

  toJSON() {
    return {
      bias: this.bias,
      weights: this.weights,
      threshold: this.threshold,
      learningRate: this.learningRate,
      data: this.data
    }
  }
}

Perceptron.defaults = {
  bias: 1,
  weights: [],
  threshold: 1,
  learningRate: 0.1,
  data: []
};

Perceptron.fromJSON = (json) => {
  return new Perceptron(json);
};