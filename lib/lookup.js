import _ from 'underscore';

/* Functions for turning sparse hashes into arrays and vice versa */
export function buildLookup(hashes) {
  // [{a: 1}, {b: 6, c: 7}] -> {a: 0, b: 1, c: 2}
  let hash = _(hashes).reduce(function(memo, hash) {
    return _(memo).extend(hash);
  }, {});
  return lookupFromHash(hash);
}

export function lookupFromHash(hash) {
  // {a: 6, b: 7} -> {a: 0, b: 1}
  let lookup = {}
    , index = 0
    ;

  for (let i in hash) {
    lookup[i] = index++;
  }

  return lookup;
}

export function toArray(lookup, hash) {
  // {a: 0, b: 1}, {a: 6} -> [6, 0]
  let array = [];
  for (let i in lookup) {
    array[lookup[i]] = hash[i] || 0;
  }
  return array;
}

export function toHash(lookup, array) {
  // {a: 0, b: 1}, [6, 7] -> {a: 6, b: 7}
  let hash = {};
  for (let i in lookup) {
    hash[i] = array[lookup[i]];
  }
  return hash;
}

export function lookupFromArray(array) {
  let lookup = {}
    // super fast loop
    , z = 0
    , i = array.length
    ;

  while (i-- > 0) {
    lookup[array[i]] = z++;
  }

  return lookup;
}
