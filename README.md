# Spiking Neural Network Model of the Primate Ventral Visual System

Developed in order to investigate the neural representation of object shape features in the ventral visual pathway, with specific emphasis on the emergence of feature binding neurons[^1] via polychronization[^2]. This model was developed using the Brian 2 simulator[^3].

## Files

The two most important files are:

* [model.py](https://github.com/patmccarthy1/spiking-PVVS-model/blob/master/model.py) - defines model architecture and includes functions to read in and present images to model
* [simulation.py](https://github.com/patmccarthy1/spiking-PVVS-model/blob/master/simulation.py) - used to run simulations by presenting images and recording spiking activity


## Model
The model architecture is illustrated below.

![diagram of model architecture](https://github.com/patmccarthy1/spiking-PVVS-model/blob/master/architecture.png)

## Author
Developed by Patrick McCarthy at Imperial College London, 2021 under the supervision of Simon Schultz, Simon Stringer and Dan Goodman.

## References

[^1]: [https://www.oftnai.org/articles/Brain_modelling_articles/Publications/Vision/2018-25960-001.pdf](https://www.oftnai.org/articles/Brain_modelling_articles/Publications/Vision/2018-25960-001.pdf)
[^2]: [https://www.mitpressjournals.org/doi/pdfplus/10.1162/089976606775093882](https://www.mitpressjournals.org/doi/pdfplus/10.1162/089976606775093882)
[^3]: [https://brian2.readthedocs.io/en/stable/index.html](https://brian2.readthedocs.io/en/stable/index.html)# Spiking neural network model of the primate ventral visual system

