export default class Convolution {
  static get defaults() {
    return {
      width: 9,
      height: null,
      inWidth: 9,
      inHeight: 9,
      depth: 3,
      stride: 3,
      padding: 0
    };
  }

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

    for (let d = 0; d < depth; d++) {
      eachFilter(d);
      const filter = this.filters[d];
      let y = -padding;
      for (let outerY = 0; outerY < outHeight; y += stride, outerY++) {
        let x = -padding;
        for (let outerX = 0; outerX < outWidth; x += stride, outerX++) {
          // convolve centered at this particular location
          const vIndex = (width * outerY) + x * depth + d;
          beforeConvolve(vIndex, outerY, outerX, d);
          for (let filterY = 0; filterY < filter.height; filterY++) {
            // coordinates in the original input array coordinates
            let innerY = y + filterY;
            for (let filterX = 0; filterX < filter.width; filterX++) {
              let innerX = x + filterX;
              if (innerY >= 0 && innerY < filter.height && innerX >= 0 && innerX < width) {
                for (let filterDepth = 0; filterDepth < filter.depth; filterDepth++) {
                  eachConvolve(
                    ((filter.width * filterY) + filterX) * filter.depth + filterDepth,
                    ((width * innerY) + innerX) * filter.depth + filterDepth
                  );
                }
              }
            }
          }
          afterConvolve(vIndex, outerX, outerY, d);
        }
      }
    }
  }
}

class Float32Array3D extends Float32Array {
  constructor(width, height, depth) {
    super(width * height * depth);
    this.width = width;
    this.height = height;
    this.depth = depth;
  }
}


var c = new Convolution();
c.build();
console.log(c.runKernel.toString());