export default class ConvolutionBuilder {
  static get defaults() {
    return {
      width: 9,
      height: null,
      inWidth: 9,
      inHeight: 9,
      filterCount: 3,
      depth: 3,
      stride: 3,
      padding: 0
    };
  }

  constructor(settings) {
    Object.assign(this, ConvolutionBuilder.defaults, settings);
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

    for(let i = 0; i < this.filterCount; i++) {
      this.filters.push(new Float32Array(this.width * this.height * this.depth));
      this.filterDeltas.push(new Float32Array(this.width * this.height * this.depth));
    }
  }

  build() {
    this.buildRunKernel();
    this.buildRunBackpropagateKernel();
  }

  buildRunKernel() {
    let fnBody = ['var weights, filterWeights'];
    this.iterateStructure({
      eachFilter: (i) => {
        fnBody.push(`filterWeights = filters[${ i }]`);
      },
      beforeConvolve: (vIndex, ax, ay, d) => {
        fnBody.push('weight = 0');
      },
      eachConvolve: (filterIndex, inputIndex) => {
        this.inputDeltas[inputIndex] = 0;
        fnBody.push(
          `weight += filterWeights[${ filterIndex }] * inputs[${ inputIndex }]`,
          `inputDeltas[${ inputIndex }] = 0`
        );
      },
      afterConvolve: (vIndex, ax, ay, d) => {
        this.biases[d] = 0;
        this.outputs[vIndex] = 0;
        fnBody.push(
          `weight += biases[${ d }]`,
          `outputs[${ vIndex }] = weight`
        );
      }
    });

    this.runKernel = new Function(
      'filters',
      'inputs',
      'outputs',
      'biases', fnBody.join(';\n  ') + ';');
  }

  run(inputs) {
    this.runKernel(this.filters, inputs, this.outputs, this.biases);
    return this.outputs;
  }

  buildRunBackpropagateKernel() {
    let fnBody = ['var chainGrad, filterWeights'];
    this.iterateStructure({
      eachFilter: (d) => {
        fnBody.push(`filterWeights = filters[${ d }]`);
      },
      beforeConvolve: (vIndex, ax, ay, d) => {
        fnBody.push(`chainGrad = chainGrads[${ vIndex }]`);
      },
      eachConvolve: (filterIndex, inputIndex) => {
        fnBody.push(
          `filterDeltas[${ filterIndex }] += inputs[${ inputIndex }] * chainGrad`,
          `inputDeltas[${ inputIndex }] += filterWeights[${ filterIndex }] * chainGrad`
        );
      },
      afterConvolve: (vIndex, ax, ay, d) => {
        this.biasDeltas[d] = 0;
        fnBody.push(`biasDeltas[${ d }] += chainGrad`);
      }
    });

    this.runBackpropagateKernel = new Function(
      'filters',
      'filterDeltas',
      'inputs',
      'inputDeltas',
      'biasDeltas',
      'chainGrads', fnBody.join(';\n  ') + ';');
  }

  runBackpropagate(inputs) {
    this.runBackpropagateKernel(this.filters, this.filterDeltas, inputs, this.inputDeltas, this.biasDeltas);
  }

  iterateStructure(options) {
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

    for (let d = 0; d < this.depth; d++) {
      eachFilter(d);
      let y = -padding;
      for (let ay = 0; ay < outHeight; y += stride, ay++) {
        let x = -padding;
        for (let ax = 0; ax < outWidth; x += stride, ax++) {
          // convolve centered at this particular location
          const vIndex = (width * ay) + x * depth + d;
          beforeConvolve(vIndex, ay, ax, d);
          for (let fy = 0; fy < height; fy++) {
            // coordinates in the original input array coordinates
            let oy = y + fy;
            for (let fx = 0; fx < width; fx++) {
              let ox = x + fx;
              if (oy >= 0 && oy < height && ox >= 0 && ox < width) {
                for (let fd = 0; fd < depth; fd++) {
                  eachConvolve(
                    ((width * fy) + fx) * depth + fd,
                    ((width * oy) + ox) * depth + fd
                  );
                }
              }
            }
          }
          afterConvolve(vIndex, ax, ay, d);
        }
      }
    }
  }
}