import { _ } from'underscore';

function testPartition(classifierConst, opts, trainOpts, trainSet, testSet) {
  let classifier = new classifierConst(opts)
    , beginTrain = Date.now()
    , trainingStats = classifier.train(trainSet, trainOpts)
    , beginTest = Date.now()
    , testStats = classifier.test(testSet)
    , endTest = Date.now()
    , stats = _(testStats).extend({
        trainTime : beginTest - beginTrain,
        testTime : endTest - beginTest,
        iterations: trainingStats.iterations,
        trainError: trainingStats.error,
        learningRate: trainOpts.learningRate,
        hidden: classifier.hiddenSizes,
        network: classifier.toJSON()
      })
    ;

  return stats;
}

export default function crossValidate(classifierConst, data, opts, trainOpts, k = 4) {
  let size = data.length / k;

  data = _(data).sortBy(function() {
    return Math.random();
  });

  let avgs = {
        error : 0,
        trainTime : 0,
        testTime : 0,
        iterations: 0,
        trainError: 0
      }
    , stats = {
        truePos: 0,
        trueNeg: 0,
        falsePos: 0,
        falseNeg: 0,
        total: 0
      }
    , misclasses = []
    , results = _.range(k).map(function(i) {
        let dclone = _(data).clone()
          , testSet = dclone.splice(i * size, size)
          , trainSet = dclone
          , result = testPartition(classifierConst, opts, trainOpts, trainSet, testSet)
          ;

        _(avgs).each(function(sum, stat) {
          avgs[stat] = sum + result[stat];
        });

        _(stats).each(function(sum, stat) {
          stats[stat] = sum + result[stat];
        });

        misclasses.push(result.misclasses);

        return result;
      })
    ;

  _(avgs).each(function(sum, i) {
    avgs[i] = sum / k;
  });

  stats.precision = stats.truePos / (stats.truePos + stats.falsePos);
  stats.recall = stats.truePos / (stats.truePos + stats.falseNeg);
  stats.accuracy = (stats.trueNeg + stats.truePos) / stats.total;

  stats.testSize = size;
  stats.trainSize = data.length - size;

  return {
    avgs: avgs,
    stats: stats,
    sets: results,
    misclasses: _(misclasses).flatten()
  };
}
