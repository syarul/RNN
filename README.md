# RNN
ML with Recurrent Neural Networks (NLP Zero to Hero) conversion to tfjs

Related to [Part 4](https://youtu.be/OuYtk9Ymut4), [Part 5](https://youtu.be/A9QVYOBjZdY) and [Part 6](https://youtu.be/ZMudJXhsUpY) by [Laurence Moroney](https://github.com/lmoroney)

# Setup

Get the sample data first

```bash
wget --no-check-certificate \
    https://storage.googleapis.com/learning-datasets/irish-lyrics-eof.txt \
    -O /tmp/irish-lyrics-eof.txt
```

```bash
npm install
node rnn.js
```
> It may take some time to complete, so it is recommended to run the code using `@tensorflow/tfjs-node-gpu` for faster execution. Alternatively, you can save the model and load it later for efficient usage.

```bash
__________________________________________________________________________________________
Layer (type)                Input Shape               Output shape              Param #   
==========================================================================================
embedding_Embedding1 (Embed [[null,15]]               [null,15,100]             269000
__________________________________________________________________________________________
bidirectional_Bidirectional [[null,15,100]]           [null,300]                301200
__________________________________________________________________________________________
dense_Dense1 (Dense)        [[null,300]]              [null,2690]               809690
==========================================================================================
Total params: 1379890
Trainable params: 1379890
Non-trainable params: 0
__________________________________________________________________________________________
I've got a bad feeling about this town on fair prince edward isle they shone delight in dublin twas praise nor her love alone that shining pike mother too god rest runaway chirping you to sleep with a baby gone alas like our youth too much i suppose fly heather no feet begin glance turns to make her drowsy too late than thady sinking funds were afford married to find the boyne from answer down to smother house oflynn fresh feet fellows are 
waiting for the wheel round the next color cant still kept hoping on the counter then rink dreaming of whiskey and sea we were
```
