var brain = require('../../lib/brain');
var assert = require('assert');

var net = new brain.NeuralNetwork(); //Create neural network
net.train([
  {input: {r:1, g:0.65, b:0},  output: {orange: 1}}, //This is orange
  {input: {r:0, g:0.54, b:0},  output: {green: 1}}, //This is green
  {input: {r:0.6, g:1, b:0.5}, output: {green: 1}}, //This is also green
  {input: {r:0.67, g:0, b:1},  output: {purple: 1}} //This is purple
]);

describe('query', function() {
  var inputs = brain.query({ count: 4, orange: { gte: 0.8 } }, net);
  assert.ok(inputs.length === 4);
  inputs.forEach(function(input) {
    assert.ok(input.r > 0);
    assert.ok(input.r <= 1);

    assert.ok(input.g >= 0);
    assert.ok(input.g <= 1);

    assert.ok(input.b >= 0);
    assert.ok(input.b <= 1);
  });
});
