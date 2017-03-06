import NeuralNetwork from './neural-network';
import Convolution from './utilities/convolution';
import Pool from './utilities/pool';
import Relu from './utilities/relu';

export default class CNN extends NeuralNetwork {
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
        const convolution = new Convolution();
        fnBody.push(`convolutionInputs = inputs[${ nodeIndex }]`);
        fnBody.push(convolution.runBody);

        fnBody.push(`poolInputs = convolutionsOutputs`);
        const pool = new Pool();
        fnBody.push(pool.runBody);

        fnBody.push(`reluInputs = poolOutputs`);
        const relu = new Relu();
        fnBody.push(relu.runBody);

        fnBody.push(`this.outputs[${ layerIndex }][${ nodeIndex }] = reluOutputs`);
      }
      fnBody.push(`output = input = this.outputs[${ layerIndex }]`);
    }
    return fnBody.join(';\n  ');
  }

  buildBackPropagateKernel() {

  }
}

const cnn = new CNN({sizes: [5, 5, 5]});
cnn.initialize([5,5,5]);
console.log(cnn.runBody);