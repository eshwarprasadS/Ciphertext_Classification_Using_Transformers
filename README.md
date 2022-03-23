# Ciphertext_Classification_Using_Transformers
Using Transformers (multi head attention encoder - decoder) to classify encrypted ciphertext with no additional context

## Method ##

Word(Token) and Positional Embeddings + Transformer

## Representation of sentence ##

Vectorized each input sentence by encoding each word in a sentence with its corresponding frequency rank; used top 20,000 words as vocabulary size. Neural Word Embeddings were used (Keras Embedding Layer) which was trained along with the final objective.

## Classifier ##

- 2 Neural Embeddings (word and position embeddings) of length 64 were fed to the Encoder, which is a Transformer_Block + timestep_avg_pool + fully_connected Layer.
- The transformer block was built with one multi-head attention layer (2 attn. heads) and one feedforward layer (64 units).
- The output from the attention layer is added to the original input and normalized and this output is fed downstream to the FF-dense layer. 
- The output from the transformer is averaged over all timesteps, which is fed to the final fully_connected layer for learning final embeddings. 
- The learning objective was to minimize cross_entropy loss between the softmax output of the network and the true_label.

## Training & Development ##

- The dev_set was used strictly to evaluate the model during the training, in each epoch and was not used during training. 
- Some key hyperparameters to be noted are : Adam Optimizer with learning rate = 1e-3, Batch Size = 64, Dropout Rate = 0.3 (For the pooled transformer output and final dense layer) and 0.1 (for attention and FF layers inside transformer block). 
- The training was done using fixed number of epochs = 20, but the best model weights (using validation accuracy on dev set) were saved, and those weights were used to make predictions on unlabeled test set.

## Requirements ##

- Pandas
- Numpy
- NLTK
- Keras
- sklearn
- Tensorflow
