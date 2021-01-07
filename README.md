# Spiking Neural Network Model of the Primate Ventral Visual System

Developed in order to investigate the neural representation of object shape features in the ventral visual pathway, with specific emphasis on the emergence of feature binding neurons<sup>[1]</sup> via polychronization<sup>[2]</sup>. This model was developed using the Brian 2 simulator<sup>[3]</sup>.

## Files

The two most important files are:

* [model.py](https://github.com/patmccarthy1/spiking-PVVS-model/blob/master/model.py) - defines model architecture and includes functions to read in and present images to model
* [simulation.py](https://github.com/patmccarthy1/spiking-PVVS-model/blob/master/simulation.py) - used to run simulations by presenting images and recording spiking activity


## Model
The model architecture is illustrated below.

![diagram of model architecture](https://github.com/patmccarthy1/spiking-PVVS-model/blob/master/architecture.png)

## Developer
Developed by Patrick McCarthy at Imperial College London, 2021 under the supervision of Simon Schultz<sup>a</sup>, Simon Stringer<sup>b</sup> and Dan Goodman<sup>c</sup>.

<sup>a</sup> <sub>Neural Coding and Neurodegenerative Disease Lab, Imperial College London</sub>\
<sup>b</sup> <sub>Centre for Theoretical Neuroscience and Artificial Intelligence, University of Oxford</sub>\
<sup>c</sup> <sub>Neural Reckoning Lab, Imperial College London</sub>
## References

[1] [https://www.oftnai.org/articles/Brain_modelling_articles/Publications/Vision/2018-25960-001.pdf](https://www.oftnai.org/articles/Brain_modelling_articles/Publications/Vision/2018-25960-001.pdf)\
[2] [https://www.mitpressjournals.org/doi/pdfplus/10.1162/089976606775093882](https://www.mitpressjournals.org/doi/pdfplus/10.1162/089976606775093882)\
[3] [https://brian2.readthedocs.io/en/stable/index.html](https://brian2.readthedocs.io/en/stable/index.html)
