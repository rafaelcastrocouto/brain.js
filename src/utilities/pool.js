export default class Pool {
  constructor(options) {
    // required
    Object.assign(this, Pool.defaults, options);
    if (this.height === null) {
      this.height = this.width;
    }

    // computed
    this.outWidth = Math.floor((this.inWidth + this.padding * 2 - this.width) / this.stride + 1);
    this.outHeight = Math.floor((this.inHeight + this.padding * 2 - this.height) / this.stride + 1);

    // store switches for x,y coordinates for where the max comes from, for each output neuron
    this.switchX = global.zeros(this.outWidth * this.outHeight * this.depth);
    this.switchY = global.zeros(this.outWidth * this.outHeight * this.depth);
    this.runKernel = null;
    this.runBackpropagateKernel = null;
    this.inputs = inputs;
    this.inputDeltas = new Array(inputs.length);
    this.outputs = outputs;
    this.outputDeltas = new Array(outputs.length);
    this.build();
  }

  build() {
    this.buildRunKernel();
    this.buildRunBackpropagateKernel();
  }

  buildRunKernel() {
    const fnBody = ['var switchIndex = 0'];
    this.iterate({
      beforePool: (outputIndex) => {
        fnBody.push(
          `var output = -99999`,
          `var winX = -1`,
          `var winY = -1`
        );
      },
      eachPool: (inputIndex) => {
        // perform max pooling and store pointers to where
        // the max came from. This will speed up backprop
        // and can help make nice visualizations in future
        fnBody.push(
          `var input = inputs[${ inputIndex }]`,
          `if (input > output) {`,
            `output = input`,
            `winX = innerX`,
            `winY = innerY`,
          `}`
        );
      },
      afterPool: (outputIndex) => {
        fnBody.push(
          `switchX[switchIndex] = winX`,
          `switchY[switchIndex] = winY`,
          `outputs[${ outputIndex }] = output`,
          `outputDeltas[${ outputIndex }] = 0`,
          `switchIndex++`
        );
      }
    });

    this.runKernel = new Function(this.runInputs(), fnBody.join(';\n  ') + ';');
  }

  runInputs() {
    return [
      'inputs',
      'outputs',
      'switchX',
      'switchY'
    ];
  }

  run() {
    this.runKernel(this.inputs, this.outputs, this.switchX, this.switchY);
    return this.outputs;
  }

  buildRunBackpropagateKernel() {
    const fnBody = ['var switchIndex = 0'];
    this.iterate({
      beforePool: (outputIndex) => {
        fnBody.push(`var outputDeltaXY = ((${ this.width } * switchY[switchIndex]) + switchX[switchIndex]) * ${ this.depth }`);
      },
      eachPool: (outputIndex, d) => {
        fnBody.push(
          `outputDeltas[outputDeltaXY + ${ d }] = outputs[${ outputIndex }]`,
          `switchIndex++`
        );
      },
      afterPool: (outputIndex) => {}
    });

    this.runKernel = new Function(this.runBackpropagateInputs(), fnBody.join(';\n  ') + ';');
  }

  runBackpropagateInputs() {
    return [
      'inputs',
      'outputs',
      'switchX',
      'switchY'
    ];
  }

  runBackpropagate() {
    this.runBackpropagateKernel(this.inputs, this.outputs, this.switchX, this.switchY);
  }

  iterate(settings) {
    const {
      padding,
      depth,
      width,
      outWidth,
      height,
      outHeight,
      stride } = this;

    const {
      beforePool,
      eachPool,
      afterPool } = settings;

    for(let d = 0; d < depth; d++) {
      let x = -padding;
      for(let outerX =0; outerX < outWidth; x += stride, outerX++) {
        let y = -padding;
        for(let outerY = 0; outerY < outHeight; y += stride, outerY++) {
          // pool centered at this particular location
          beforePool(outerX * outerY * d);
          for(let filterX = 0; filterX < width; filterX++) {
            for(let filterY = 0; filterY < height; filterY++) {
              let innerY = y + filterY;
              let innerX = x + filterX;
              if(
                innerY < 0
                && innerY >= height
                && innerX < 0
                && innerX >= width
              ) continue;
              eachPool(innerX * innerY * d, d);
            }
          }
          afterPool(outerX * outerY * d);
        }
      }
    }
  }
}

Pool.defaults = {
  width: 9,
  height: null,
  inWidth: 9,
  inHeight: 9,
  depth: 3,
  stride: 2,
  padding: 0
};