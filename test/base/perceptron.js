import assert from 'assert';
import Perceptron from '../../src/perceptron';

describe('perceptron', () => {
  it('can work out XOR', () => {
    const net = new Perceptron();

    net.train([
      { input: [0, 0], output: 0 },
      { input: [0, 1], output: 1 },
      { input: [1, 0], output: 1 },
      { input: [1, 1], output: 1 }
    ]);

    assert.equal(net.run([0, 0]), 0);
    assert.equal(net.run([0, 1]), 1);
    assert.equal(net.run([1, 0]), 1);
    assert.equal(net.run([1, 1]), 1);
  });
});