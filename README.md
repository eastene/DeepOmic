# DeepOmic

A project to test various architectures of deep autoencoders for reducing the dimensionality of 'omics data.

## Running
* It is recommended to first convert your data from comma or tab delimited files to tfrecords if your data is large (more than a couple hundred rows) to speed up training. If your data is in csv format you can run `csv2tfrecord.py` in the data directory to convert.
* To run DeepOmic with default parameters:
  1. `cd model`
  2. `python deepomicmodel.py`

## Flags
A number of model hyperparameters can be set via flags defined in `flags.py` in the model directory. Select flags of importance are:
* `sparsity_lambda` (default 0.0) sets a sparsity constraint on the loss function, changes autoencoder to a sparse autoencoder (currently unused due to conflicts with denoising loss function) (Glorot et. al. 2011)
* `emphasis_alpha` (default 0.1) sets emphasis of corrupted features for loss function of denoising autoencoder (Vincent et.al 2010)
* `emphasis_beta` (default 0.9) sets emphasis of uncorrupted features for loss function of denoising autoencoder (Vincent et.al 2010)

---
**This work was presented at the COPDGene Deep Learning Workshop 2018.**

## References:
* Glorot, Xavier, Antoine Bordes, and Yoshua Bengio. "Deep sparse rectifier neural networks." Proceedings of the fourteenth international conference on artificial intelligence and statistics. 2011.
* Vincent, Pascal, et al. "Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion." Journal of machine learning research 11.Dec (2010): 3371-3408.
