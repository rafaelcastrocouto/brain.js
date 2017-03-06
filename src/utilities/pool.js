import zeros from './zeros';

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
    this.switchX = zeros(this.outWidth * this.outHeight * this.depth);
    this.switchY = zeros(this.outWidth * this.outHeight * this.depth);
    this.runKernel = null;
    this.runBackpropagateKernel = null;
    this.inputs = zeros(this.width * this.height);
    this.inputDeltas = zeros(this.width * this.height);
    this.outputs = zeros(this.outWidth * this.outHeight);
    this.outputDeltas = zeros(this.outWidth * this.outHeight);
    this.build();
  }

  build() {
    this.buildRunKernel();
    this.buildRunBackpropagateKernel();
  }

  get runBody() {
    const fnBody = ['var poolSwitchIndex = 0'];
    this.iterate({
      beforePool: (outputIndex) => {
        fnBody.push(
          `var poolOutput = -99999`,
          `var poolWinX = -1`,
          `var poolWinY = -1`
        );
      },
      eachPool: (inputIndex) => {
        // perform max pooling and store pointers to where
        // the max came from. This will speed up backprop
        // and can help make nice visualizations in future
        fnBody.push(
          `var poolInput = poolInputs[${ inputIndex }]`,
          `if (poolInput > poolOutput) {`,
          `  poolOutput = poolInput`,
          `  poolWinX = poolInnerX`,
          `  poolWinY = poolInnerY`,
          `}`
        );
      },
      afterPool: (outputIndex) => {
        fnBody.push(
          `poolSwitchX[poolSwitchIndex] = poolWinX`,
          `poolSwitchY[poolSwitchIndex] = poolWinY`,
          `poolOutputs[${ outputIndex }] = poolOutput`,
          `poolOutputDeltas[${ outputIndex }] = 0`,
          `poolSwitchIndex++`
        );
      }
    });
    return fnBody.join(';\n  ') + ';'
  }

  buildRunKernel() {
    this.runKernel = new Function(this.runInputs(), this.runBody);
  }

  runInputs() {
    return [
      'poolInputs',
      'poolOutputs',
      'poolSwitchX',
      'poolSwitchY'
    ];
  }

  run() {
    this.runKernel(this.inputs, this.outputs, this.switchX, this.switchY);
    return this.outputs;
  }

  get runBackpropagateBody() {
    const fnBody = ['var poolSwitchIndex = 0'];
    this.iterate({
      beforePool: (outputIndex) => {
        fnBody.push(`var poolOutputDeltaXY = ((${ this.width } * poolSwitchY[poolSwitchIndex]) + poolSwitchX[poolSwitchIndex]) * ${ this.depth }`);
      },
      eachPool: (outputIndex, d) => {
        fnBody.push(
          `poolOutputDeltas[poolOutputDeltaXY + ${ d }] = poolOutputs[${ outputIndex }]`,
          `poolSwitchIndex++`
        );
      },
      afterPool: (outputIndex) => {}
    });
    return fnBody.join(';\n  ') + ';';
  }

  buildRunBackpropagateKernel() {
    this.runKernel = new Function(this.runBackpropagateInputs(), this.runBackpropagateBody);
  }

  runBackpropagateInputs() {
    return [
      'poolInputs',
      'poolOutputs',
      'poolSwitchX',
      'poolSwitchY'
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