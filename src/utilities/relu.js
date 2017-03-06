import zeros from './zeros';

export default class Relu {
  constructor(inputs, outputs) {
    this.runKernel = null;
    this.runBackpropagateKernel = null;
    this.inputs = zeros(this.width * this.height);
    this.outputs = zeros(this.width * this.height);
    this.inputDeltas = new Array(inputs.length);
    this.outputDeltas = new Array(outputs.length);
    this.build();
  }

  build() {
    this.buildRunKernel();
    this.buildRunBackpropagateKernel();
  }

  get runBody() {
    const fnBody = [

    ];
    this.iterate({
      each: (i) => {
        fnBody.push(
          `reluOutputs[${ i }] = reluInputs[${ i }] < 0 ? 0 : reluInputs[${ i }]`,
          `reluInputDeltas[${ i }] = 0`,
          `reluOutputDeltas[${ i }] = 0`
        );
      }
    });
    return fnBody.join(';\n  ') + ';';
  }

  buildRunKernel() {
    this.runKernel = new Function(this.runInputs, this.runBody);
  }

  run() {
    this.runKernel(this.input, this.inputDeltas, this.output, this.outputDeltas);
  }

  get runBackpropagateBody() {
    const fnBody = [];
    this.iterate({
      each: (i) => {
        fnBody.push(`reluInputDeltas[${ i }] = reluOutput[${ i }] <= 0 ? 0 : reluOutputDeltas[${ i }]`);
      }
    });
    return fnBody.join(';\n  ') + ';';
  }

  buildRunBackpropagateKernel() {
    this.runBackpropagateKernel = new Function(this.runBackpropagateInputs, this.runBackpropagateBody);
  }

  runBackpropagate() {
    this.runBackpropagateKernel(this.input, this.inputDeltas, this.output, this.outputDeltas);
  }

  iterate(settings) {
    const { input } = this;
    for(let i = 0; i < input.length; i++) {
      settings.each(i);
    }
  }
}

Relu.runInputs = [
  'reluInputs',
  'reluInputDeltas',
  'reluOutputs',
  'reluOutputDeltas'
];

Relu.runBackpropagateInputs = [
  'reluInputs',
  'reluInputDeltas',
  'reluOutputs',
  'reluOutputDeltas'
];