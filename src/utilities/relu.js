export default class Relu {
  constructor(inputs, outputs) {
    this.runKernel = null;
    this.runBackpropagateKernel = null;
    this.inputs = inputs;
    this.outputs = outputs;
    this.inputDeltas = new Array(inputs.length);
    this.outputDeltas = new Array(outputs.length);
    this.build();
  }

  build() {
    this.buildRunKernel();
    this.buildRunBackpropagateKernel();
  }

  static runInputs = [
    'inputs',
    'inputDeltas',
    'outputs',
    'outputDeltas'
  ];

  get runBody() {
    const fnBody = [];
    this.iterate({
      each: (i) => {
        fnBody.push(
          `outputs[${ i }] = inputs[${ i }] < 0 ? 0 : inputs[${ i }]`,
          `inputDeltas[${ i }] = 0`,
          `outputDeltas[${ i }] = 0`
        );
      }
    });
    return fnBody.join(';\n  ') + ';';
  }

  buildRunKernel() {
    this.runKernel = new Function(
      this.runInputs, this.runBody);
  }

  run() {
    this.runKernel(this.input, this.inputDeltas, this.output, this.outputDeltas);
  }

  static runBackpropagateInputs = [
    'inputs',
    'inputDeltas',
    'outputs',
    'outputDeltas'
  ];

  get runBackpropagateBody() {
    const fnBody = [];
    this.iterate({
      each: (i) => {
        fnBody.push(`inputDeltas[${ i }] = output[${ i }] <= 0 ? 0 : outputDeltas[${ i }]`);
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