pyRealFakeProducer scrapes iRealPro's 1460 forumn for chord sequences, and trains a transformer encoder/decoder to predict subsequent embeddings. It's still work in progress, but can learn 2-5-1s or turn around's within the fitst 20 songs

It's currently packaged with Cuda in mind, and will be until the hyperparameters approach an ideal final state for my setup. A Dockerfile will be added soon, the current package list for the system and pip are outlined in flake.nix

the generate_starts_with... function can be used for inference. Future iterations will remove the eval loops that were necessary to find hyperparameters that didn't always decode to E7#11/G#, and the process should batch inputs
