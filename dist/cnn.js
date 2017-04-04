'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

var _neuralNetwork = require('./neural-network');

var _neuralNetwork2 = _interopRequireDefault(_neuralNetwork);

var _convolution = require('./utilities/convolution');

var _convolution2 = _interopRequireDefault(_convolution);

var _pool = require('./utilities/pool');

var _pool2 = _interopRequireDefault(_pool);

var _relu = require('./utilities/relu');

var _relu2 = _interopRequireDefault(_relu);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

function _possibleConstructorReturn(self, call) { if (!self) { throw new ReferenceError("this hasn't been initialised - super() hasn't been called"); } return call && (typeof call === "object" || typeof call === "function") ? call : self; }

function _inherits(subClass, superClass) { if (typeof superClass !== "function" && superClass !== null) { throw new TypeError("Super expression must either be null or a function, not " + typeof superClass); } subClass.prototype = Object.create(superClass && superClass.prototype, { constructor: { value: subClass, enumerable: false, writable: true, configurable: true } }); if (superClass) Object.setPrototypeOf ? Object.setPrototypeOf(subClass, superClass) : subClass.__proto__ = superClass; }

var CNN = function (_NeuralNetwork) {
  _inherits(CNN, _NeuralNetwork);

  function CNN() {
    _classCallCheck(this, CNN);

    return _possibleConstructorReturn(this, (CNN.__proto__ || Object.getPrototypeOf(CNN)).apply(this, arguments));
  }

  _createClass(CNN, [{
    key: 'runInput',

    /**
     *
     * @param input
     * @returns {*}
     */
    value: function runInput(input) {
      throw new Error('not yet implemented');
      this.outputs[0] = input; // set output state of input layer

      var output = null;
      for (var layer = 1; layer <= this.outputLayer; layer++) {
        for (var node = 0; node < this.sizes[layer]; node++) {
          var weights = this.weights[layer][node];
          //TODO: CNN logic here
          this.outputs[layer][node] = convol;
        }
        output = input = this.outputs[layer];
      }
      return output;
    }
  }, {
    key: 'buildBackPropagateKernel',
    value: function buildBackPropagateKernel() {}
  }, {
    key: 'runBody',
    get: function get() {
      var fnBody = ['this.outputs[0] = input', 'var output = null', 'var inputs = []'];

      for (var layerIndex = 1; layerIndex <= this.outputLayer; layerIndex++) {
        for (var nodeIndex = 0; nodeIndex < this.sizes[layerIndex]; nodeIndex++) {
          var convolution = new _convolution2.default();
          fnBody.push(convolution.runBody);

          var pool = new _pool2.default();
          fnBody.push(pool.runBody);

          var relu = new _relu2.default();
          fnBody.push(relu.runBody);

          fnBody.push('this.outputs[' + layerIndex + '][' + nodeIndex + '] = output');
        }
        fnBody.push('output = input = this.outputs[' + layerIndex + ']');
      }
      return output;
    }
  }]);

  return CNN;
}(_neuralNetwork2.default);

exports.default = CNN;


console.log(new CNN().runBody);
//# sourceMappingURL=cnn.js.map