export default class Input {
  constructor(options) {
    Object.assign(this, Input.defaults, options);
    if (this.height === null) {
      this.height = this.width;
    }

    this.weights = new Float32Array(this.width * this.height * this.depth);
    this.deltas = new Float32Array(this.width * this.height * this.depth);
  }
}

Input.defaults = {
  width: 10,
  height: null,
  depth: 1
};