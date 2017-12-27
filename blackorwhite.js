const codeBox = document.getElementById('code-box');
const progressBox = document.getElementById('progress-box');
const swatch = document.querySelectorAll('.swatch');
const trainingBox = document.getElementById('training-box');
const trainingMessage = document.getElementById('training-message');
const testingBox = document.getElementById('testing-box');
const testBox = document.getElementById('test-box');
const wcagSwatchBox = document.getElementById('wcag-swatch-box');
const nnSwatch = document.getElementById('nn-swatch');
const wcagSwatch = document.getElementById('wcag-swatch');
const yiqSwatch = document.getElementById('yiq-swatch');

progressBox.style.display = 'none';
testingBox.style.display = 'none';
codeBox.style.display = 'none';
wcagSwatchBox.style.display = 'none';
testBox.style.display = 'none';

const utils = {
  randomColor: function() {
    return {
      r: Math.round(Math.random() * 255),
      g: Math.round(Math.random() * 255),
      b: Math.round(Math.random() * 255)
    };
  },
  toRgb: function(color) {
    return 'rgb(' + color.r + ',' + color.g + ',' + color.b + ')';
  },
  normalize: function(color) {
    return {
      r: color.r / 255,
      g: color.g / 255,
      b: color.b / 255
    };
  }
};

const trainer = {
  currentColor: utils.randomColor(),
  data: [],
  pickSwatch: function(color) {
    const result = {
      input: utils.normalize(this.currentColor),
      output: { black: color === 'black' ? 1: 0 }
    };
    this.data.push(result);

    this.changeColor();

    // show the 'Train network' button after we've selected a few entries
    if (this.data.length === 5) {
      testBox.style.display = '';
    }
  },

  changeColor: function() {
    this.currentColor = utils.randomColor();
    swatch.forEach(el => el.style.backgroundColor = utils.toRgb(this.currentColor));
  },

  trainNetwork: function() {
    trainingBox.style.display = 'none';
    progressBox.style.display = '';

    if(window.Worker100) {
      const worker = new Worker('training-worker.js');
      worker.onmessage = this.onMessage;
      worker.onerror = this.onError;
      worker.postMessage(JSON.stringify(this.data));
    }
    else {
      const net = new brain.NeuralNetworkGPU();
      net.train(this.data, {
        iterations: 100
      });
      tester.show(net);
    }
  },

  onMessage: function(event) {
    const data = JSON.parse(event.data);
    if(data.type === 'progress') {
      trainer.showProgress(data);
    }
    else if(data.type === 'result') {
      const net = new brain.NeuralNetworkGPU().fromJSON(data.net);
      tester.show(net);
    }
  },

  onError: function(event) {
    trainingMessage.innerText = 'error training network: ' + event.message;
  },

  showProgress: function(progress) {
    const completed = progress.iterations / trainer.iterations * 100;
    trainingMessage.style.width = completed + '%';
  }
};

const tester = {
  show: function(net) {
    progressBox.style.display = 'none';
    runNetwork = net;
    //runNetwork.name = 'runNetwork'; // for view code later
    this.testRandom();
    testingBox.style.display = '';
  },

  testRandom: function() {
    this.testColor(utils.randomColor());
  },

  testColor: function(color) {
    swatch.forEach(el => el.backgroundColor = utils.toRgb(color));
    const colorNormalized = utils.normalize(color);
    nnSwatch.style.color = nnColor(colorNormalized);
    wcagSwatch.style.color = wcagColor(colorNormalized);
    yiqSwatch.style.color = yiqColor(colorNormalized);
  },

  viewCode: function(type) {
    if(type === 'nn' && !$('#nn-swatch-box').hasClass('selected')) {
      $('#code-header').text('neural network code:');
      var code = 'var textColor = ' + nnColor.toString()
                  + '\n\nvar runNetwork = ' + runNetwork.toString();
      $('#code').text(code);
      $('.swatch-box').removeClass('selected');
      $('#nn-swatch-box').addClass('selected');
      $('#code-box').show();
    } else if(type === 'wcag' && !$('#wcag-swatch-box').hasClass('selected')) {
      $('#code-header').text('luminosity algorithm code:');
      var code = 'var textColor = ' + wcagColor.toString()
                  + '\n\nvar contrast = ' + contrast.toString()
                  + '\n\nvar luminosity = ' + luminosity.toString();
      $('#code').text(code);
      $('.swatch-box').removeClass('selected');
      $('#wcag-swatch-box').addClass('selected');
      $('#code-box').show();
    } else if(type === 'yiq' && !$('#yiq-swatch-box').hasClass('selected')) {
      $('#code-header').text('YIQ formula code:');
      var code = 'var textColor = ' + yiqColor.toString();

      $('#code').text(code);
      $('.swatch-box').removeClass('selected');
      $('#yiq-swatch-box').addClass('selected');
      $('#code-box').show();
    } else {
      $('#code-box').hide();
      $('.swatch-box').removeClass('selected');
    }
  }
};


/* these functions are outside so we can just call toString() for 'view code'*/
function nnColor(bgColor) {
  const output = runNetwork(bgColor);
  if (output.black > .5) {
    return 'black';
  }
  return 'white';
}

function wcagColor(bgColor) {
  if(contrast(bgColor, {r: 1, g: 1, b: 1})
      > contrast(bgColor, {r: 0, g: 0, b: 0}))
    return 'white';
  return 'black';
}

function luminosity(color) {
  const r = color.r, g = color.g, b = color.b;
  const red = (r <= 0.03928) ? r / 12.92: Math.pow(((r + 0.055)/1.055), 2.4);
  const green = (g <= 0.03928) ? g / 12.92: Math.pow(((g + 0.055)/1.055), 2.4);
  const blue = (b <= 0.03928) ? b / 12.92: Math.pow(((b + 0.055)/1.055), 2.4);

  return 0.2126 * red + 0.7152 * green + 0.0722 * blue;
}

function contrast(color1, color2) {
  const lum1 = luminosity(color1);
  const lum2 = luminosity(color2);
  if (lum1 > lum2) {
    return (lum1 + 0.05) / (lum2 + 0.05);
  }
  return (lum2 + 0.05) / (lum1 + 0.05);
}

function yiqColor(bgColor) {
  const r = bgColor.r * 255;
  const g = bgColor.g * 255;
  const b = bgColor.b * 255;
  const yiq = (r * 299 + g * 587 + b * 114) / 1000;
  return (yiq >= 128) ? 'black': 'white';
}
