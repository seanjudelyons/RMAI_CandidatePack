## Redmarble.ai

For convience I have made use of Google Colab to present the solutions.  

## Solution 1

In this solution I am using a simple Embedding and Flatten layer.  


```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```
The model had an ```accuracy: 0.9979``` and a ```val_accuracy: 0.8520```.  However, the model is not able to make predictions satisfyingly.

## Solution 2


In this solution I am transfer learning with just a single Dense layer to collate the embeddings.  Each sentence is encoded using [Sentence Transformers](https://pypi.org/project/sentence-transformers/), specifically the [BERT](https://arxiv.org/abs/1810.04805) model.

```bash
pip install -U sentence-transformers
```
I also semantic encoded the labels using [GloVe](https://nlp.stanford.edu/projects/glove/), where the number 1 was replaced with the semantic vectors for String "yes", and 0 with the semantic vectors of String "no".

## Visualisation
The results of solution 2 can be downloaded.  The ```meta-data.tsv``` and the ```vecs-data.tsv``` can be downloaded and uploaded directly to the [TensorFlow Projector](https://projector.tensorflow.org/) to see how the model performs.  The sentences in the visualisation are of the actual delay data and have their labels attached so its easier to recognise if the model is making the correct predictions.

![Alt Text](https://media.giphy.com/media/9IAk3QRHHitdlYcNmy/giphy.gif)

## TensorFlow Lite
I used a base [BERT](https://docs.google.com/spreadsheets/d/14QplCdTCDwEmTqrn1LH4yrbKvdogK4oQvYO1K1aPR5M/edit#gid=0) model so the model can be incorporated in applications.  
