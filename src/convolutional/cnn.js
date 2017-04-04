import NeuralNetwork from '../neural-network';
import Convolution from './formula/convolution';
import Pool from './formula/pool';
import Relu from './formula/relu';

export default class CNN extends NeuralNetwork {
  constructor(options = {}) {
    super(options);
    Object.assign(this, CNN.defaults, options);
  }
  /**
   *
   * @param input
   * @returns {*}
   */
  runInput(input) {
    throw new Error('not yet implemented');
    this.outputs[0] = input;  // set output state of input layer

    let output = null;
    for (let layer = 1; layer <= this.outputLayer; layer++) {
      for (let node = 0; node < this.sizes[layer]; node++) {
        let weights = this.weights[layer][node];
        //TODO: CNN logic here
        this.outputs[layer][node] = convol;
      }
      output = input = this.outputs[layer];
    }
    return output;
  }

  get runBody() {
    const fnBody = [
      'this.outputs[0] = input',
      'var output = null',
      'var inputs = []'
    ];

    for (let layerIndex = 1; layerIndex <= this.outputLayer; layerIndex++) {
      for (let nodeIndex = 0; nodeIndex < this.sizes[layerIndex]; nodeIndex++) {
        const convolution = this.convolutions[layerIndex][nodeIndex];
        fnBody.push(`convolutionInputs = inputs[${ nodeIndex }]`);
        fnBody.push(convolution.runBody);

        const relu = this.relus[layerIndex][nodeIndex];
        fnBody.push(`reluInputs = convolutionOutputs`);
        fnBody.push(relu.runBody);

        const pool = this.pools[layerIndex][nodeIndex];
        fnBody.push(`poolInputs = reluOutputs`);
        fnBody.push(pool.runBody);

        fnBody.push(`this.outputs[${ layerIndex }][${ nodeIndex }] = poolOutputs`);
      }
      fnBody.push(`output = input = this.outputs[${ layerIndex }]`);
    }

    return fnBody.join(';\n  ');
  }

  buildBackPropagateKernel() {

  }

  initialize() {
    if (this.input.constructor === Function) {
      this.input = this.input()
    }
    this.hiddenLayers.forEach((hiddenLayer, i) => {
      const input = (i > 0 ? this.hiddenLayers[i - 1] : this.input);
      if (hiddenLayer.constructor === Function) {
        this.hiddenLayers[i] = hiddenLayer(input);
      } else {
        this.hiddenLayers[i] = new hiddenLayer(input);
      }
    });
  }
}

CNN.defaults = {
  input: () => new Input(35, 35, 3),
  hiddenLayers: [
    (input) => new Convolution(input, {
      depth: 1,
      filters: 6,
      padding: 1,
      stride: 1
    }),
    Relu,
    (input) => new Pool(input, {})
  ]
};

const cnn = new CNN().runBody.toString();
console.log(cnn);