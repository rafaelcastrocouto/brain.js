'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _possibleConstructorReturn(self, call) { if (!self) { throw new ReferenceError("this hasn't been initialised - super() hasn't been called"); } return call && (typeof call === "object" || typeof call === "function") ? call : self; }

function _inherits(subClass, superClass) { if (typeof superClass !== "function" && superClass !== null) { throw new TypeError("Super expression must either be null or a function, not " + typeof superClass); } subClass.prototype = Object.create(superClass && superClass.prototype, { constructor: { value: subClass, enumerable: false, writable: true, configurable: true } }); if (superClass) Object.setPrototypeOf ? Object.setPrototypeOf(subClass, superClass) : subClass.__proto__ = superClass; }

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var Convolution = function () {
  _createClass(Convolution, null, [{
    key: 'defaults',
    get: function get() {
      return {
        width: 9,
        height: null,
        inWidth: 9,
        inHeight: 9,
        depth: 3,
        stride: 3,
        padding: 0
      };
    }
  }]);

  function Convolution(settings) {
    _classCallCheck(this, Convolution);

    Object.assign(this, Convolution.defaults, settings);
    this.runKernel = null;
    this.runBackpropagateKernel = null;
    this.filters = [];
    this.filterDeltas = [];
    this.biasDeltas = [];
    this.biases = [];
    this.inputDeltas = [];
    this.outputs = [];

    if (this.height === null) {
      this.height = this.width;
    }
    this.outWidth = Math.floor((this.width + this.padding * 2 - this.width) / this.stride + 1);
    this.outHeight = Math.floor((this.height + this.padding * 2 - this.height) / this.stride + 1);

    for (var i = 0; i < this.depth; i++) {
      this.filters.push(new Float32Array3D(this.width, this.height, this.depth));
      this.filterDeltas.push(new Float32Array3D(this.width, this.height, this.depth));
    }

    this.build();
  }

  _createClass(Convolution, [{
    key: 'build',
    value: function build() {
      this.buildRunKernel();
      this.buildRunBackpropagateKernel();
    }
  }, {
    key: 'buildRunKernel',
    value: function buildRunKernel() {
      var _this = this;

      var fnBody = ['var weights, filterWeights'];
      this.iterateStructure({
        eachFilter: function eachFilter(i) {
          fnBody.push('filterWeights = filters[' + i + ']');
        },
        beforeConvolve: function beforeConvolve(vIndex, ax, ay, d) {
          fnBody.push('weight = 0');
        },
        eachConvolve: function eachConvolve(filterIndex, inputIndex) {
          _this.inputDeltas[inputIndex] = 0;
          fnBody.push('weight += filterWeights[' + filterIndex + '] * inputs[' + inputIndex + ']', 'inputDeltas[' + inputIndex + '] = 0');
        },
        afterConvolve: function afterConvolve(vIndex, ax, ay, d) {
          _this.biases[d] = 0;
          _this.outputs[vIndex] = 0;
          fnBody.push('weight += biases[' + d + ']', 'outputs[' + vIndex + '] = weight');
        }
      });

      this.runKernel = new Function('filters', 'inputs', 'outputs', 'biases', fnBody.join(';\n  ') + ';');
    }
  }, {
    key: 'run',
    value: function run(inputs) {
      this.runKernel(this.filters, inputs, this.outputs, this.biases);
      return this.outputs;
    }
  }, {
    key: 'buildRunBackpropagateKernel',
    value: function buildRunBackpropagateKernel() {
      var _this2 = this;

      var fnBody = ['var chainGrad, filterWeights'];
      this.iterateStructure({
        eachFilter: function eachFilter(d) {
          fnBody.push('filterWeights = filters[' + d + ']');
        },
        beforeConvolve: function beforeConvolve(vIndex, ax, ay, d) {
          fnBody.push('chainGrad = chainGrads[' + vIndex + ']');
        },
        eachConvolve: function eachConvolve(filterIndex, inputIndex) {
          fnBody.push('filterDeltas[' + filterIndex + '] += inputs[' + inputIndex + '] * chainGrad', 'inputDeltas[' + inputIndex + '] += filterWeights[' + filterIndex + '] * chainGrad');
        },
        afterConvolve: function afterConvolve(vIndex, ax, ay, d) {
          _this2.biasDeltas[d] = 0;
          fnBody.push('biasDeltas[' + d + '] += chainGrad');
        }
      });

      this.runBackpropagateKernel = new Function('filters', 'filterDeltas', 'inputs', 'inputDeltas', 'biasDeltas', 'chainGrads', fnBody.join(';\n  ') + ';');
    }
  }, {
    key: 'runBackpropagate',
    value: function runBackpropagate(inputs) {
      this.runBackpropagateKernel(this.filters, this.filterDeltas, inputs, this.inputDeltas, this.biasDeltas);
    }
  }, {
    key: 'iterateStructure',
    value: function iterateStructure(options) {
      var width = this.width,
          height = this.height,
          outWidth = this.outWidth,
          outHeight = this.outHeight,
          depth = this.depth,
          stride = this.stride,
          padding = this.padding;
      var eachFilter = options.eachFilter,
          beforeConvolve = options.beforeConvolve,
          eachConvolve = options.eachConvolve,
          afterConvolve = options.afterConvolve;


      for (var d = 0; d < depth; d++) {
        eachFilter(d);
        var filter = this.filters[d];
        var y = -padding;
        for (var outerY = 0; outerY < outHeight; y += stride, outerY++) {
          var x = -padding;
          for (var outerX = 0; outerX < outWidth; x += stride, outerX++) {
            // convolve centered at this particular location
            var vIndex = width * outerY + x * depth + d;
            beforeConvolve(vIndex, outerY, outerX, d);
            for (var filterY = 0; filterY < filter.height; filterY++) {
              // coordinates in the original input array coordinates
              var innerY = y + filterY;
              for (var filterX = 0; filterX < filter.width; filterX++) {
                var innerX = x + filterX;
                if (innerY >= 0 && innerY < filter.height && innerX >= 0 && innerX < width) {
                  for (var filterDepth = 0; filterDepth < filter.depth; filterDepth++) {
                    eachConvolve((filter.width * filterY + filterX) * filter.depth + filterDepth, (width * innerY + innerX) * filter.depth + filterDepth);
                  }
                }
              }
            }
            afterConvolve(vIndex, outerX, outerY, d);
          }
        }
      }
    }
  }]);

  return Convolution;
}();

exports.default = Convolution;

var Float32Array3D = function (_Float32Array) {
  _inherits(Float32Array3D, _Float32Array);

  function Float32Array3D(width, height, depth) {
    _classCallCheck(this, Float32Array3D);

    var _this3 = _possibleConstructorReturn(this, (Float32Array3D.__proto__ || Object.getPrototypeOf(Float32Array3D)).call(this, width * height * depth));

    _this3.width = width;
    _this3.height = height;
    _this3.depth = depth;
    return _this3;
  }

  return Float32Array3D;
}(Float32Array);

var c = new Convolution();
c.build();
console.log(c.runKernel.toString());
//# sourceMappingURL=convolution.js.map