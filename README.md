pyRealFakeProducer scrapes iRealPro's 1460 fakebook songs for chord sequences, and trains a transformer encoder/decoder to predict subsequent embeddings. It's still work in progress, but can learn 2-5-1s or turn around's within the first 20 songs. It might be overfitting past 5% of the database, but with my current hardware it takes a while to reach that point. It still hasn't been fully trained with teacher forcing.

It's currently packaged with Cuda in mind, and will be until the hyperparameters approach an ideal final state for my setup. A Dockerfile will be added soon, the current package list for the system and pip are outlined in flake.nix

the generate\_starts\_with... function can be used for inference. Future iterations will remove the eval loops that were necessary to find hyperparameters that didn't always decode to E7#11/G#, and the process should batch inputs

This project is still in the prototype stage. The most recent git commit is not necessarily the best at making predictions. Please discuss if you find better hyperparameters
