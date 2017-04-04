'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

var _zeros = require('./zeros');

var _zeros2 = _interopRequireDefault(_zeros);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var Pool = function () {
  function Pool(options) {
    _classCallCheck(this, Pool);

    // required
    Object.assign(this, Pool.defaults, options);
    if (this.height === null) {
      this.height = this.width;
    }

    // computed
    this.outWidth = Math.floor((this.inWidth + this.padding * 2 - this.width) / this.stride + 1);
    this.outHeight = Math.floor((this.inHeight + this.padding * 2 - this.height) / this.stride + 1);

    // store switches for x,y coordinates for where the max comes from, for each output neuron
    this.switchX = (0, _zeros2.default)(this.outWidth * this.outHeight * this.depth);
    this.switchY = (0, _zeros2.default)(this.outWidth * this.outHeight * this.depth);
    this.runKernel = null;
    this.runBackpropagateKernel = null;
    this.inputs = (0, _zeros2.default)(this.width * this.height);
    this.inputDeltas = (0, _zeros2.default)(this.width * this.height);
    this.outputs = (0, _zeros2.default)(this.outWidth * this.outHeight);
    this.outputDeltas = (0, _zeros2.default)(this.outWidth * this.outHeight);
    this.build();
  }

  _createClass(Pool, [{
    key: 'build',
    value: function build() {
      this.buildRunKernel();
      this.buildRunBackpropagateKernel();
    }
  }, {
    key: 'buildRunKernel',
    value: function buildRunKernel() {
      this.runKernel = new Function(this.runInputs(), this.runBody);
    }
  }, {
    key: 'runInputs',
    value: function runInputs() {
      return ['reluInputs', 'reluOutputs', 'switchX', 'switchY'];
    }
  }, {
    key: 'run',
    value: function run() {
      this.runKernel(this.inputs, this.outputs, this.switchX, this.switchY);
      return this.outputs;
    }
  }, {
    key: 'buildRunBackpropagateKernel',
    value: function buildRunBackpropagateKernel() {
      this.runKernel = new Function(this.runBackpropagateInputs(), this.runBackpropagateBody);
    }
  }, {
    key: 'runBackpropagateInputs',
    value: function runBackpropagateInputs() {
      return ['inputs', 'outputs', 'switchX', 'switchY'];
    }
  }, {
    key: 'runBackpropagate',
    value: function runBackpropagate() {
      this.runBackpropagateKernel(this.inputs, this.outputs, this.switchX, this.switchY);
    }
  }, {
    key: 'iterate',
    value: function iterate(settings) {
      var padding = this.padding,
          depth = this.depth,
          width = this.width,
          outWidth = this.outWidth,
          height = this.height,
          outHeight = this.outHeight,
          stride = this.stride;
      var beforePool = settings.beforePool,
          eachPool = settings.eachPool,
          afterPool = settings.afterPool;


      for (var d = 0; d < depth; d++) {
        var x = -padding;
        for (var outerX = 0; outerX < outWidth; x += stride, outerX++) {
          var y = -padding;
          for (var outerY = 0; outerY < outHeight; y += stride, outerY++) {
            // pool centered at this particular location
            beforePool(outerX * outerY * d);
            for (var filterX = 0; filterX < width; filterX++) {
              for (var filterY = 0; filterY < height; filterY++) {
                var innerY = y + filterY;
                var innerX = x + filterX;
                if (innerY < 0 && innerY >= height && innerX < 0 && innerX >= width) continue;
                eachPool(innerX * innerY * d, d);
              }
            }
            afterPool(outerX * outerY * d);
          }
        }
      }
    }
  }, {
    key: 'runBody',
    get: function get() {
      var fnBody = ['var switchIndex = 0'];
      this.iterate({
        beforePool: function beforePool(outputIndex) {
          fnBody.push('var reluOutput = -99999', 'var winX = -1', 'var winY = -1');
        },
        eachPool: function eachPool(inputIndex) {
          // perform max pooling and store pointers to where
          // the max came from. This will speed up backprop
          // and can help make nice visualizations in future
          fnBody.push('var reluInput = reluInputs[' + inputIndex + ']', 'if (reluInput > reluOutput) {', 'reluOutput = reluInput', 'winX = innerX', 'winY = innerY', '}');
        },
        afterPool: function afterPool(outputIndex) {
          fnBody.push('switchX[switchIndex] = winX', 'switchY[switchIndex] = winY', 'reluOutputs[' + outputIndex + '] = reluOutput', 'reluOutputDeltas[' + outputIndex + '] = 0', 'switchIndex++');
        }
      });
      return fnBody.join(';\n  ') + ';';
    }
  }, {
    key: 'runBackpropagateBody',
    get: function get() {
      var _this = this;

      var fnBody = ['var switchIndex = 0'];
      this.iterate({
        beforePool: function beforePool(outputIndex) {
          fnBody.push('var outputDeltaXY = ((' + _this.width + ' * switchY[switchIndex]) + switchX[switchIndex]) * ' + _this.depth);
        },
        eachPool: function eachPool(outputIndex, d) {
          fnBody.push('outputDeltas[outputDeltaXY + ' + d + '] = outputs[' + outputIndex + ']', 'switchIndex++');
        },
        afterPool: function afterPool(outputIndex) {}
      });
      return fnBody.join(';\n  ') + ';';
    }
  }]);

  return Pool;
}();

exports.default = Pool;


Pool.defaults = {
  width: 9,
  height: null,
  inWidth: 9,
  inHeight: 9,
  depth: 3,
  stride: 2,
  padding: 0
};
//# sourceMappingURL=pool.js.map