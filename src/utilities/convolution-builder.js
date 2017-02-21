export default class ConvolutionBuilder {
  constructor(settings) {
    this.value = [];

    this.run = null;
    this.runBackpropagate = null;
    this.buildRun();
    this.buildRunBackpropagate();
  }

  buildRun() {
    let fnBody = 'var a;';
    this.iterateStructure({
      width: 0,
      height: 0,
      stride: 0,
      padding: 0,
      eachDepth: (d) => {
        fnBody += `var filter = filters[${ d }];`;
        return {width: 0, height: 0, depth: 0};
      },
      beforeConvolve: (vIndex, ax, ay, d) => {
        fnBody += 'a = 0;';
      },
      eachConvolve: (filterIndex, weightIndex) => {
        fnBody += `a += filter[${ filterIndex }] * inputs[${ weightIndex }];`;
      },
      afterConvolve: (vIndex, ax, ay, d) => {
        fnBody += `a += biases[${ d }];`;
        fnBody += `outputs[${ vIndex }] = a;`;
      }
    });

    this.run = new Function('filters', 'inputs', 'outputs', 'biases', fnBody);
  }

  iterateStructure(options) {
    const {width, height, outWidth, outHeight, depth, eachDepth, beforeConvolve, eachConvolve, afterConvolve, stride, padding} = options;
    for (let d = 0; d < depth; d++) {
      const filterSize = eachDepth(d);
      let y = -padding;
      for (let ay = 0; ay < outHeight; y += stride, ay++) {
        let x = -padding;
        for (let ax = 0; ax < outWidth; x += stride, ax++) {
          // convolve centered at this particular location
          const vIndex = (width * ay) + x * depth + d;
          beforeConvolve(vIndex, ay, ax, d);
          for (let fy = 0; fy < filterSize.height; fy++) {
            // coordinates in the original input array coordinates
            let oy = y + fy;
            for (let fx = 0; fx < filterSize.width; fx++) {
              let ox = x + fx;
              if (oy >= 0 && oy < height && ox >= 0 && ox < width) {
                for (let fd = 0; fd < filterSize.depth; fd++) {
                  eachConvolve(
                    ((filterSize.width * fy) + fx) * filterSize.depth + fd,
                    ((width * oy) + ox) * depth + fd
                  );
                }
              }
            }
          }
          afterConvolve(vIndex, ax, ay, d);
        }
      }
    }
  }

  buildRunBackpropagate(options) {
    let fnBody = 'var dw, chainGrad;';
    this.iterateStructure({
      width: 0,
      height: 0,
      stride: 0,
      padding: 0,
      eachDepth: (d) => {
        fnBody += `var filter = filters[${ d }];`;
        return {width: 0, height: 0, depth: 0};
      },
      beforeConvolve: (vIndex, ax, ay, d) => {
        fnBody += `chainGrad = dw[${ vIndex }]`;
        fnBody += 'dw = 0;';
      },
      eachConvolve: (filterIndex, weightIndex) => {
        fnBody += `dw += filter[${ filterIndex }] * chainGrad;`;
        fnBody += `dw += inputs[${ weightIndex }] * chainGrad;`;
      },
      afterConvolve: (vIndex, ax, ay, d) => {
        fnBody += `dw += biases[${ d }];`;
        fnBody += `outputs[${ vIndex }] = dw;`;
      }
    });

    this.runBackpropagate = new Function('filters', 'inputs', 'outputs', 'biases', fnBody);
  }
}