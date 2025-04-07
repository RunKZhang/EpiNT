# EpiNT

Official implementations of paper "Cross-Modal Epileptic Signal Harmonization: Frequency Domain Mapping Quantization for Pre-training a Unified Neurophysiological Transformer"

## Description

We propose EpiNT, a novel Transformer-based pre-trained model designed for unified analysis of both EEG and iEEG in epilepsy. 
The pre-training paradigm integrates masked autoencoders (MAE) with vector quantization (VQ). 
We propose a frequency domain mapping quantizer, which explicitly discretizes neurophysiological signals in the frequency domain, and requires no train.

## Getting Started

### Dependencies

* PyTorch 
* Aim (you can use other monitor package also)
* HuggingFace

### How to Run?

To run the code, you should firstly run the ```generate_downstream_datasets.py``` and ```generate_pretrain_dataset.py``` under ```scripts```.
* To pre-train the model, you should run the bash file ```run_pretrain.sh```.
* To finetune the model, please run the bash file ```run_finetune.sh```.

## Help

Since we have no rights to publish the datasets we used, we provide the download links in file ```dataset_list.py```. 
All the datasets we used are public, you can download them from the link provided.

## Authors

If you have any questions, please feel free to contact: 

Runkai Zhang (e-mail: [271013216[replace with at]qq.com]())


## License

This project is licensed under the [MIT] License - see the LICENSE.md file for details.

## Citation

If you like this repo, please rise me a star.

We will update the citation info once the paper is accepted.