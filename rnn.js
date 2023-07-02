/* eslint-disable camelcase */
import tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-node'
import { readFileSync } from 'fs'

import { pad_sequences } from 'array-sequence-utils'
import Tokenizer from 'text-tokenizer-utils/keras_preprocessing_text.js'

const tokenizer = new Tokenizer()

const learning_rate = 0.01 // adam optimizer learning rate

const data = readFileSync('./tmp/irish-lyrics-eof.txt', 'utf-8')

const corpus = data.toLowerCase().split('\n')

await tokenizer.fit_on_texts(corpus)
const word_index = await tokenizer.word_index()
// "+1" refers to the additional pad token that will be added to the sequence, which is why we include the "+1"
const total_words = Object.entries(word_index).length + 1

console.log(Object.entries(word_index).filter(a => a[1] < 15))
console.log(total_words)

// generate the n_gram_sequence
let input_sequences = []
for (const line of corpus) {
  const [token_list] = await tokenizer.texts_to_sequences([line])
  for (let i = 1; i < token_list.length; i++) {
    const n_gram_sequence = token_list.slice(0, i + 1)
    input_sequences.push(n_gram_sequence)
  }
}

// pad sequences
const max_sequence_len = Math.max(...input_sequences.map(x => x.length))
input_sequences = pad_sequences(input_sequences, max_sequence_len, 'pre')

// ensure the inputs is converted to tensor before being fed to the model
const xs = tf.tensor2d(input_sequences.map(sequence => sequence.slice(0, -1)))

// the labels need to be categorical and one hot encoded
const labels = input_sequences.map(sequence => sequence.slice(-1)[0])
const ys = tf.oneHot(labels, total_words)

console.log(word_index['in'])
console.log(word_index['the'])
console.log(word_index['town'])
console.log(word_index['of'])
console.log(word_index['athy'])
console.log(word_index['one'])
console.log(word_index['jeremy'])
console.log(word_index['lanigan'])

xs.slice([6], [1]).print()
ys.slice([6], [1]).print()

xs.slice([5], [1]).print()
ys.slice([5], [1]).print()

const model = tf.sequential()
model.add(tf.layers.embedding({ inputDim: total_words, outputDim: 100, inputLength: max_sequence_len - 1 }))
model.add(tf.layers.bidirectional({ layer: tf.layers.lstm({ units: 150 }) }))
model.add(tf.layers.dense({ units: total_words, activation: 'softmax' }))
const optimizer = tf.train.adam(learning_rate)
model.compile({ loss: 'categoricalCrossentropy', optimizer, metrics: ['accuracy'] })

model.summary()

await model.fit(xs, ys, {
  epochs: 100,
  verbose: 2,
})

// await model.save('file://irish_lyrics')

// const loadModel = await tf.loadLayersModel('file://irish_lyrics/model.json')

let seed_text = 'I\'ve got a bad feeling about this'

const next_words = 100

for (let _ = 0; _ < next_words; _++) {
  const [token_list] = await tokenizer.texts_to_sequences([seed_text])
  const padded_token_list = pad_sequences([token_list], max_sequence_len - 1, 'pre')
  const predicted = model.predict(tf.tensor2d(padded_token_list, [1, max_sequence_len - 1])).argMax(-1).dataSync()[0]
  let output_word = ''
  for (const [word, index] of Object.entries(word_index)) {
    if (index === predicted) {
      output_word = word
      break
    }
  }
  seed_text += ' ' + output_word
}

console.log(seed_text)

tokenizer.close()
