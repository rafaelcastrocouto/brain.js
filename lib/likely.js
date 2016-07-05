export default function (input, net) {
  let output = net.run(input)
    , maxProp = null
    , maxValue = -1
    ;

  for (let prop in output) {
    if (output.hasOwnProperty(prop)) {
      var value = output[prop];
      if (value > maxValue) {
        maxProp = prop;
        maxValue = value
      }
    }
  }
  
  return maxProp;
};
