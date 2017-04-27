/**
 * add {left} and {right} matrix weights into {into}
 * @param {Matrix} product
 * @param {Matrix} left
 * @param {Matrix} right
 */
export default function forwardPropagate(product, left, right) {
  const body = ['  '];
  for(let i = 0; i < left.weights.length; i++) {
    body.push(
      'productWeights[i] = leftWeights[i] + rightWeights[i]',
      'productDeltas[i] = 0'
    );
  }

  body.push('return productWeights');

  return new Function(['productWeights', 'leftWeights', 'rightWeights', 'productDeltas'], body.join(';\n  '));
}

export default function backwardPropagate(product, left, right) {
  const body = [];
  for(let i = 0; i < product.deltas.length; i++) {
    body.push(
      'leftDeltas[i] = productDeltas[i]',
      'rightDeltas[i] = productDeltas[i]'
    );
  }
}

export default function add(product, left, right) {
  const body = [];
  for(let i = 0; i < left.weights.length; i++) {
    body.push('productWeights[i] = leftWeights[i] + rightWeights[i]');
    product.deltas[i] = 0;
  }
}