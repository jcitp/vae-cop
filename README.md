# Monte Carlo and variational autoencoder for the conserved order-parameter Ising model

## Logic

The file cop.cpp contains a simple Monte Carlo routine to generate samples of the conserverd order parameter Ising model. You can store the raw configurations alongside with the order parameter and the temperature.

The python script vaeCOP.py creates a variational autoencoder using keras and trains and validates it with the Monte Carlo samples generated with cop.cpp.

The Jupyter notebook AnalyseVAE-COP.ipynb uses the trained keras models to do the phase-transition estimation.


## References

The files accompany our [machine learning blog](http://www.cmt-qo.phys.ethz.ch/cmt-qo-news/2018/02/from-pca-to-variational-autoencoders.html)

The details on the conserved order parameter Ising model you find in [Lei Wang's paper](https://dx.doi.org/10.1103/PhysRevB.94.195105)

And the variational autoencoder is heavily based on the [Keras blog](https://blog.keras.io/building-autoencoders-in-keras.html)
