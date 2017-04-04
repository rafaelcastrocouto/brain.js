import Filter from './filter';
import Bias from './bias';

export default class Convolution {
  constructor(input, options) {
    this.input = input;
    Object.assign(this, Convolution.defaults, options);
    this.runKernel = null;
    this.runBackpropagateKernel = null;
    this.filters = [];
    this.biases = [];
    this.outputs = [];

    const { padding, stride, width, height, depth } = this;
    this.width = Math.floor((input.width + padding * 2 - width) / stride + 1);
    this.height = Math.floor((input.height + padding * 2 - height) / stride + 1);

    for(let i = 0; i < depth; i++) {
      this.filters.push(new Filter(width, height, depth));
      this.biases.push(new Bias());
    }

    this.build();
  }

  build() {
    this.buildRunKernel();
    this.buildRunBackpropagateKernel();
  }

  get runBody() {
    let fnBody = [
      'var convolutionWeights',
      'var convolutionFilterWeights'
    ];
    this.iterate({
      eachFilter: (i) => {
        fnBody.push(`convolutionFilterWeights = convolutionFilters[${ i }]`);
      },
      beforeConvolve: (outputIndex, ax, ay, d) => {
        fnBody.push('convolutionWeight = 0');
      },
      eachConvolve: (filterIndex, inputIndex) => {
        fnBody.push(
          `convolutionWeight += convolutionFilterWeights[${ filterIndex }] * convolutionInputs[${ inputIndex }]`,
          `convolutionInputDeltas[${ inputIndex }] = 0`
        );
      },
      afterConvolve: (outputIndex, ax, ay, d) => {
        this.biases.weights[d] = 0;
        this.outputs[outputIndex] = 0;
        fnBody.push(
          `convolutionWeight += convolutionBiases[${ d }]`,
          `convolutionOutputs[${ outputIndex }] = convolutionWeight`,
          `convolutionOutputDeltas[${ outputIndex }] = 0`
        );
      }
    });
    return fnBody.join(';\n  ') + ';';
  }

  buildRunKernel() {
    this.runKernel = new Function(
      'convolutionFilters',
      'convolutionInputs',
      'convolutionOutputs',
      'convolutionBiases', this.runBody);
  }

  run(inputs) {
    this.runKernel(this.filters, inputs, this.outputs, this.biases);
    return this.outputs;
  }

  get runBackPropagateBody() {
    let fnBody = [
      'var convolutionChainGradient',
      'var convolutionFilterWeights'
    ];
    this.iterate({
      eachFilter: (d) => {
        fnBody.push(`convolutionFilterWeights = convolutionFilters[${ d }]`);
      },
      beforeConvolve: (vIndex, ax, ay, d) => {
        fnBody.push(`convolutionChainGradient = convolutionChainGradients[${ vIndex }]`);
      },
      eachConvolve: (filterIndex, inputIndex) => {
        fnBody.push(
          `convolutionFilterDeltas[${ filterIndex }] += convolutionInputs[${ inputIndex }] * convolutionChainGradient`,
          `convolutionInputDeltas[${ inputIndex }] += convolutionFilterWeights[${ filterIndex }] * convolutionChainGradient`
        );
      },
      afterConvolve: (vIndex, ax, ay, d) => {
        this.biasDeltas[d] = 0;
        fnBody.push(`convolutionBiasDeltas[${ d }] += convolutionChainGradient`);
      }
    });
    return fnBody.join(';\n  ') + ';';
  }

  buildRunBackpropagateKernel() {
    this.runBackpropagateKernel = new Function(
      'convolutionFilters',
      'convolutionFilterDeltas',
      'convolutionInputs',
      'convolutionInputDeltas',
      'convolutionBiasDeltas',
      'convolutionChainGradients', this.runBackPropagateBody);
  }

  runBackpropagate(inputs) {
    this.runBackpropagateKernel(this.filters, inputs, this.inputDeltas, this.biasDeltas);
  }

  iterate(options) {
    const {
      input,
      width,
      height,
      depth,
      stride,
      padding } = this;

    const {
      width: inputWidth,
      height: inputHeight
    } = input;

    const {
      eachFilter,
      beforeConvolve,
      eachConvolve,
      afterConvolve } = options;

    this.filters.forEach((filter, filterIndex) => {
      eachFilter(filterIndex);
      let y = -padding;
      console.log(height,);
      for (let outerY = 0; outerY < height; y += stride, outerY++) {
        let x = -padding;
        for (let outerX = 0; outerX < width; x += stride, outerX++) {
          // convolve centered at this particular location
          const outputIndex = (inputWidth * outerY) + x * depth + filterIndex;
          console.log(filter);
          beforeConvolve(outputIndex, outerY, outerX, filterIndex);

          filter.iterate((filterX, filterY, filterDepth) => {
            console.log(filterX, filterY, filterDepth);
            let innerY = y + filterY;
            let innerX = x + filterX;
            if (
              innerY < 0
              && innerY >= filter.height
              && innerX < 0
              && innerX >= inputWidth
            ) return;

            eachConvolve(
              ((filter.width * filterY) + filterX) * filter.depth + filterDepth,
              ((inputWidth * innerY) + innerX) * filter.depth + filterDepth
            );
          });
          afterConvolve(outputIndex, outerX, outerY, filterIndex);
        }
      }
    });
  }
}

Convolution.defaults = {
  depth: 3,
  stride: 3,
  padding: 1,
  width: 9,
  height: 9
};

var conv = new Convolution({ height: 27, width: 27 });
conv.build();
console.log(conv.runKernel.toString());