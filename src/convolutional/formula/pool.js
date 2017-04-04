import zeros from '../../utilities/zeros';

export default class Pool {
  constructor(input, options) {
    this.input = input;
    const inputWidth = input.width;
    const inputHeight = input.height;

    Object.assign(this, Pool.defaults, options);

    // computed
    this.width = Math.floor((inputWidth + this.padding * 2 - this.width) / this.stride + 1);
    this.height = Math.floor((inputHeight + this.padding * 2 - this.height) / this.stride + 1);

    // store switches for x,y coordinates for where the max comes from, for each output neuron
    this.switchX = zeros(this.width * this.height * this.depth);
    this.switchY = zeros(this.width * this.height * this.depth);
    this.runKernel = null;
    this.runBackpropagateKernel = null;
    this.outputs = zeros(this.width * this.height);
    this.outputDeltas = zeros(this.width * this.height);
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
      height,
      stride } = this;

    const {
      width: inputWidth,
      height: inputHeight
    } = this.input;

    const {
      beforePool,
      eachPool,
      afterPool } = settings;

    for(let d = 0; d < depth; d++) {
      let x = -padding;
      for(let outerX =0; outerX < inputWidth; x += stride, outerX++) {
        let y = -padding;
        for(let outerY = 0; outerY < inputHeight; y += stride, outerY++) {
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
  depth: 3,
  stride: 2,
  padding: 0
};