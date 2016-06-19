var retro = require('./retro');

module.exports = function(command, net) {
  var i = 0;
  var max = command.count;
  var results = [];
  var keys = Object.keys(net.layers[net.layers.length]);

  for (;i < max; i++) {
    var result = {};
    keys.forEach(function(key) {
      if (command.hasOwnProperty(key)) {
        if (command.hasOwnProperty('gte')) {
          return result[key] = Math.random() * (1 - command.gte) + command.gte;
        }
        if (command.hasOwnProperty('gt')) {
          return result[key] = Math.random() * (1 - command.gt + 0.001) + command.gt + 0.001;
        }
        if (command.hasOwnProperty('lte')) {
          return result[key] = Math.random() * command.lte;
        }
        if (command.hasOwnProperty('lt')) {
          return result[key] = Math.random() * (command.lte - 0.001);
        }
        if (command.hasOwnProperty('eq')) {
          return result[key] = command.eq;
        }

        throw new Error('keys gte, gt, lte, lt, or eq must be set');
      } else {
        result[key] = 0;
      }
    });
    results.push(retro(result, net));
  }

  return results;
};
