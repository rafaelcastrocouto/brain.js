export default class Filter {
  constructor(width, height, depth) {
    const length = this.length = width * height * depth;
    this.weights = new Float32Array(length);
    this.deltas = new Float32Array(length);
    this.width = width;
    this.height = height;
    this.depth = depth;
  }

  iterate(fn) {
    for (let y = 0; y < this.height; y++) {
      for (let x = 0; x < this.width; x++) {
        for (let d = 0; d < this.depth; d++) {
          fn(x, y, d);
        }
      }
    }
  }
}