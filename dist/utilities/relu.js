'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

var _zeros = require('./zeros');

var _zeros2 = _interopRequireDefault(_zeros);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var Relu = function () {
  function Relu(inputs, outputs) {
    _classCallCheck(this, Relu);

    this.runKernel = null;
    this.runBackpropagateKernel = null;
    this.inputs = (0, _zeros2.default)(this.width * this.height);
    this.outputs = (0, _zeros2.default)(this.width * this.height);
    this.inputDeltas = new Array(inputs.length);
    this.outputDeltas = new Array(outputs.length);
    this.build();
  }

  _createClass(Relu, [{
    key: 'build',
    value: function build() {
      this.buildRunKernel();
      this.buildRunBackpropagateKernel();
    }
  }, {
    key: 'buildRunKernel',
    value: function buildRunKernel() {
      this.runKernel = new Function(this.runInputs, this.runBody);
    }
  }, {
    key: 'run',
    value: function run() {
      this.runKernel(this.input, this.inputDeltas, this.output, this.outputDeltas);
    }
  }, {
    key: 'buildRunBackpropagateKernel',
    value: function buildRunBackpropagateKernel() {
      this.runBackpropagateKernel = new Function(this.runBackpropagateInputs, this.runBackpropagateBody);
    }
  }, {
    key: 'runBackpropagate',
    value: function runBackpropagate() {
      this.runBackpropagateKernel(this.input, this.inputDeltas, this.output, this.outputDeltas);
    }
  }, {
    key: 'iterate',
    value: function iterate(settings) {
      var input = this.input;

      for (var i = 0; i < input.length; i++) {
        settings.each(i);
      }
    }
  }, {
    key: 'runBody',
    get: function get() {
      var fnBody = [];
      this.iterate({
        each: function each(i) {
          fnBody.push('outputs[' + i + '] = inputs[' + i + '] < 0 ? 0 : inputs[' + i + ']', 'inputDeltas[' + i + '] = 0', 'outputDeltas[' + i + '] = 0');
        }
      });
      return fnBody.join(';\n  ') + ';';
    }
  }, {
    key: 'runBackpropagateBody',
    get: function get() {
      var fnBody = [];
      this.iterate({
        each: function each(i) {
          fnBody.push('inputDeltas[' + i + '] = output[' + i + '] <= 0 ? 0 : outputDeltas[' + i + ']');
        }
      });
      return fnBody.join(';\n  ') + ';';
    }
  }]);

  return Relu;
}();

exports.default = Relu;


Relu.runInputs = ['inputs', 'inputDeltas', 'outputs', 'outputDeltas'];

Relu.runBackpropagateInputs = ['inputs', 'inputDeltas', 'outputs', 'outputDeltas'];
//# sourceMappingURL=relu.js.map