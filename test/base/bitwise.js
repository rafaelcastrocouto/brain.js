import assert from 'assert';
import brain from '../../src';

let wiggle = 0.1;

function isAround(actual, expected) {
  if (actual > (expected + wiggle)) return false;
  if (actual < (expected - wiggle)) return false;
  return true;
}

function testBitwise(data, activation, op) {
  let net = new brain.NeuralNetwork();
  let res = net.train(data, {
    errorThresh: 0.003,
    activation
  });

  console.log(res);

  data.forEach(d => {
    const actual = net.run(d.input);
    const expected = d.output;
    assert.ok(isAround(actual, expected), `failed to train "${op}" - expected: ${expected}, actual: ${actual}`);
  });
}

describe('bitwise functions sync training', () => {
  const not = [
    { input: [0], output: [1] },
    { input: [1], output: [0] }
  ];
  const xor = [
    { input: [0, 0], output: [0] },
    { input: [0, 1], output: [1] },
    { input: [1, 0], output: [1] },
    { input: [1, 1], output: [0] }
  ];
  const or = [
    { input: [0, 0], output: [0] },
    { input: [0, 1], output: [1] },
    { input: [1, 0], output: [1] },
    { input: [1, 1], output: [1] }
  ];
  const and = [
    { input: [0, 0], output: [0] },
    { input: [0, 1], output: [0] },
    { input: [1, 0], output: [0] },
    { input: [1, 1], output: [1] }
  ];
  describe('sigmoid', () => {
    it('NOT function', () => {
      testBitwise(not, 'sigmoid', 'not');
    });

    it('XOR function', () => {
      testBitwise(xor, 'sigmoid', 'xor');
    });

    it('OR function', () => {
      testBitwise(or, 'sigmoid', 'or');
    });

    it('AND function', () => {
      testBitwise(and, 'sigmoid', 'and');
    });
  });

  describe('tanh', () => {
    it('NOT function', () => {
      testBitwise(not, 'tanh', 'not');
    });

    it('XOR function', () => {
      testBitwise(xor, 'tanh', 'xor');
    });

    it('OR function', () => {
      testBitwise(or, 'tanh', 'or');
    });

    it('AND function', () => {
      testBitwise(and, 'tanh', 'and');
    });
  });

  describe('relu', () => {
    it('NOT function', () => {
      testBitwise(not, 'relu', 'not');
    });

    it('XOR function', () => {
      testBitwise(xor, 'relu', 'xor');
    });

    it('OR function', () => {
      testBitwise(or, 'relu', 'or');
    });

    it('AND function', () => {
      testBitwise(and, 'relu', 'and');
    });
  });

  describe('leaky-relu', () => {
    it('NOT function', () => {
      testBitwise(not, 'leaky-relu', 'not');
    });

    it('XOR function', () => {
      testBitwise(xor, 'leaky-relu', 'xor');
    });

    it('OR function', () => {
      testBitwise(or, 'leaky-relu', 'or');
    });

    it('AND function', () => {
      testBitwise(and, 'leaky-relu', 'and');
    });
  });
});