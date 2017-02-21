import NeuralNetwork from './neural-network';

export default class CNN extends NeuralNetwork {
  /**
   *
   * @param input
   * @returns {*}
   */
  runInput(input) {
    this.outputs[0] = input;  // set output state of input layer

    let output = null;
    for (let layer = 1; layer <= this.outputLayer; layer++) {
      for (let node = 0; node < this.sizes[layer]; node++) {
        let weights = this.weights[layer][node];

        let sum = this.biases[layer][node];
        for (let k = 0; k < weights.length; k++) {
          sum += weights[k] * input[k];
        }
        this.outputs[layer][node] = 1 / (1 + Math.exp(-sum));
      }
      output = input = this.outputs[layer];
    }
    return output;
  }

  forward(V) {
    const vStrideX = V.strideX || 0;
    const vStrideY = V.strideY || 0;
    const stride = this.stride || 0;

    for(let depthIndex = 0; depthIndex < this.outDepth; depthIndex++) {
      const filter = this.filters[depthIndex];
      let y = -this.padding || 0;
      for(let outputYIndex = 0; outputYIndex < this.outStrideY; y += stride, outputYIndex++) {
        let x = -this.padding || 0;
        for(let outputXIndex = 0; outputXIndex < this.outStrideX; x += stride, outputXIndex++) {
          let sum = 0;
          for(let filterYIndex = 0; filterYIndex < filter.strideY; filterYIndex++) {
            let inputYIndex = y + filterYIndex; // coordinates in the original input array coordinates
            for(let filterXIndex = 0; filterXIndex < filter.strideX; filterXIndex++) {
              let inputXIndex = x + filterXIndex;
              if(inputYIndex < 0 && inputYIndex >= vStrideY && inputXIndex < 0 && inputXIndex >= vStrideX) continue;
              for(let filterDepthIndex = 0; filterDepthIndex < filter.depth; filterDepthIndex++) {
                sum += filter.w[
                    ((filter.strideX * filterYIndex) + filterXIndex)
                    * filter.depth + filterDepthIndex
                  ]
                  * V.w[
                    ((vStrideX * inputYIndex) + inputXIndex)
                    * V.depth + filterDepthIndex];
              }
            }
          }
          sum += this.biases.w[d];
          A.set(outputXIndex, outputYIndex, depthIndex, sum);
        }
      }
    }
    this.out_act = A;
    return this.out_act;
  }
}

export const ConvolutionalNeuralNetwork = CNN;

CNN.defaults = {
  filterX: 5,
  filterY: 5,
  filterZ: 3,
  filterStrideX: 1,
  filterStrideY: 1,
  padding: 0
};

const initialVec = [
  11,12,13,14,15,17,18,19,
  21,22,23,24,25,27,28,29,
  31,32,33,34,35,37,38,39,
  41,42,43,44,45,47,48,49,
  51,52,53,54,55,57,58,59,
  61,62,63,64,65,67,68,69,
  71,72,73,74,75,77,78,79,
  81,82,83,84,85,87,88,89,
  91,92,93,94,95,97,98,99
];
function iterate(vec1, vec2) {
  for (let y = 0; y < CNN.defaults.filterY; y += CNN.defaults.stride) {
    for (let x = 0; x < CNN.defaults.filterX; x += CNN.defaults.stride) {

    }
  }
}
function convolve(vec1, vec2) {
  let result = [];
  for (let i = 0; i < vec2.length ; i++){
    result.push(vec1[0] * vec2[i]);
  }
  let displacement = 0;
  for (let i = 1; i < vec1.length ; i++){
    for (let j = 0; j < vec2.length ; j++){
      let resultIndex = displacement + j;
      if (resultIndex !== result.length) {
        result[resultIndex] += vec1[i] * vec2[j];
      } else {
        result.push(vec1[i] * vec2[j]);
      }
    }
    displacement++;
  }
  return result;
}
