export default class Convolution {
  constructor(settings) {
    Object.assign(this, Convolution.defaults, settings);
    this.runKernel = null;
    this.runBackpropagateKernel = null;
    this.filters = [];
    this.filterDeltas = [];
    this.biasDeltas = [];
    this.biases = [];
    this.inputDeltas = [];
    this.outputs = [];

    if (this.height === null) {
      this.height = this.width;
    }
    this.outWidth = Math.floor((this.width + this.padding * 2 - this.width) / this.stride + 1);
    this.outHeight = Math.floor((this.height + this.padding * 2 - this.height) / this.stride + 1);

    for(let i = 0; i < this.depth; i++) {
      this.filters.push(new Float32Array3D(this.width, this.height, this.depth));
      this.filterDeltas.push(new Float32Array3D(this.width, this.height, this.depth));
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
        this.inputDeltas[inputIndex] = 0;
        fnBody.push(
          `convolutionWeight += convolutionFilterWeights[${ filterIndex }] * convolutionInputs[${ inputIndex }]`,
          `convolutionInputDeltas[${ inputIndex }] = 0`
        );
      },
      afterConvolve: (outputIndex, ax, ay, d) => {
        this.biases[d] = 0;
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
    this.runBackpropagateKernel(this.filters, this.filterDeltas, inputs, this.inputDeltas, this.biasDeltas);
  }

  iterate(options) {
    const {
      width,
      height,
      outWidth,
      outHeight,
      depth,
      stride,
      padding } = this;

    const {
      eachFilter,
      beforeConvolve,
      eachConvolve,
      afterConvolve } = options;

    for (let d = 0; d < depth; d++) {
      eachFilter(d);
      const filter = this.filters[d];
      let y = -padding;
      for (let outerY = 0; outerY < outHeight; y += stride, outerY++) {
        let x = -padding;
        for (let outerX = 0; outerX < outWidth; x += stride, outerX++) {
          // convolve centered at this particular location
          const outputIndex = (width * outerY) + x * depth + d;
          beforeConvolve(outputIndex, outerY, outerX, d);
          for (let filterY = 0; filterY < filter.height; filterY++) {
            // coordinates in the original input array coordinates
            let innerY = y + filterY;
            for (let filterX = 0; filterX < filter.width; filterX++) {
              let innerX = x + filterX;
              if (
                innerY < 0
                && innerY >= filter.height
                && innerX < 0
                && innerX >= width
              ) continue;

              for (let filterDepth = 0; filterDepth < filter.depth; filterDepth++) {
                eachConvolve(
                  ((filter.width * filterY) + filterX) * filter.depth + filterDepth,
                  ((width * innerY) + innerX) * filter.depth + filterDepth
                );
              }
            }
          }
          afterConvolve(outputIndex, outerX, outerY, d);
        }
      }
    }
  }
}

Convolution.defaults = {
  width: 9,
  height: null,
  inWidth: 9,
  inHeight: 9,
  depth: 3,
  stride: 3,
  padding: 0
};

class Float32Array3D extends Float32Array {
  constructor(width, height, depth) {
    super(width * height * depth);
    this.width = width;
    this.height = height;
    this.depth = depth;
  }
}
