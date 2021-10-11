# Spiking Neural Network Model of the Primate Ventral Visual System

Biologically realistic model of a network of neurons in the primate ventral visual pathway. This was developed to investigate the emergence of feature selectivity for forming representations of shape, and to test a newly proposed solution to feature binding: hierarchical binding by polychronization <sup>[1,2]</sup>.

## Model Architecture

Images are processed by a Gabor filter set, and pixel values of the filtered images determine the mean firing rate of Poisson neurons, acting as an input layer. The rest of the model consists of 4 layers of conductance-based LIF neurons, containing excitatory and inhibitory neurons in a 4:1 ratio. Neurons are connected in a topologically corresponding manner to facilitate the development of retinotopic maps. Neurons are connected with feedforward, feedback, recurrent and lateral (inhibitory) conductance-based synapses. Conduction delays between neurons are Gaussian distributed, and an STDP learning rule is implemented in synapses between excitatory LIF neurons, encouraging the emergence of polychronous neuronal groups. 

The model architecture is illustrated below.

<p align="center">
<img src="https://github.com/patmccarthy1/spiking-PVVS-model/blob/master/misc/architecture.png" data-canonical-src="https://github.com/patmccarthy1/spiking-PVVS-model/blob/master/misc/architecture.png" height="500" />
</p>

## Requirements

* Python v3.7.7
* MATLAB R2021a
* Brian v2.3.0.2
* NumPy v1.19.1
* SciPy v1.6.2
* OpenCV v3.4.2
* Matplotlib v3.3.1

## Model Use

After installing the dependencies mentioned above, clone this Git repository to your local drive. The most important files and folders are:

* [model.py](https://github.com/patmccarthy1/spiking-PVVS-model/blob/master/model.py) - defines model architecture and includes functions present images to model
* [simulation.py](https://github.com/patmccarthy1/spiking-PVVS-model/blob/master/simulation.py) - defines simulation specifications
* [simulation.sh](https://github.com/patmccarthy1/spiking-PVVS-model/blob/master/simulation.sh) - used to run simulations 
* [analysis](https://github.com/patmccarthy1/spiking-PVVS-model/blob/master/analysis) - contains a number of MATLAB scripts and Python notebooks to perform data analysis and plotting

## Development
Developed by Patrick McCarthy at Imperial College London, 2021 under the supervision of Simon Schultz<sup>a</sup>, Simon Stringer<sup>b</sup> and Dan Goodman<sup>c</sup>.

<sup>a</sup> <sub>Neural Coding Lab, Imperial College London</sub>\
<sup>b</sup> <sub>Centre for Theoretical Neuroscience and Artificial Intelligence, University of Oxford</sub>\
<sup>c</sup> <sub>Neural Reckoning Lab, Imperial College London</sub>
## References

[1] [https://www.oftnai.org/articles/Brain_modelling_articles/Publications/Vision/2018-25960-001.pdf](https://www.oftnai.org/articles/Brain_modelling_articles/Publications/Vision/2018-25960-001.pdf)\
[2] [https://www.mitpressjournals.org/doi/pdfplus/10.1162/089976606775093882](https://www.mitpressjournals.org/doi/pdfplus/10.1162/089976606775093882)\
[3] [https://brian2.readthedocs.io/en/stable/index.html](https://brian2.readthedocs.io/en/stable/index.html)
