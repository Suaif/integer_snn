# Full Integer Arithmetic Online Training of Spiking Neural Network

This is a private repo for the PyTorch implementation for my master thesis **Full Integer Arithmetic Online Training of Spiking Neural Network**

Spiking Neural Networks (SNNs) offer substantial advantages for neuromorphic computing
due to their biological plausibility and energy efficiency. However, traditional training methods
such as Backpropagation Through Time (BPTT) and Real Time Recurrent Learning (RTRL)
are computationally intensive and difficult to implement efficiently on neuromorphic hardware.
This thesis proposes a novel integer-only, online training algorithm tailored specifically for SNNs,
leveraging a mixed-precision approach and integer-only arithmetic to significantly enhance computational efficiency and reduce memory usage.
The developed learning algorithm integrates gradient-based methods inspired by Real-Time
Recurrent Learning (RTRL) and spatiotemporal backpropagation, using local eligibility traces to
approximate gradients. The algorithm employs integer-only arithmetic, replacing costly floatingpoint operations with computationally efficient bit-shift operations, making it particularly suitable for deployment on specialized neuromorphic hardware. Three error propagation strategies
(Feedback, Final, and Direct) are evaluated, with the Direct approach emerging as optimal for
integer-based training due to its stability and lower computational complexity. Additionally,
the integer training algorithm is successfully extended beyond fully connected SNNs to Convolutional Spiking Neural Networks (CSNNs) and Recurrent Spiking Neural Networks (RSNNs),
demonstrating versatility across architectures.
Extensive experiments are conducted on two widely used datasets: MNIST for static vision
tasks and the Spiking Heidelberg Digits (SHD) dataset for temporal neuromorphic tasks. Results
indicate that mixed-precision configurations, particularly those using 16-bit shadow weights and
8- or 12-bit inference weights, achieve comparable or superior accuracy relative to full-precision
floating-point implementations, reducing memory usage by over 60% and computational energy
by more than an order of magnitude. Despite limitations in multi-layer training and ultralow precision configurations, the method achieves robust performance, matching or exceeding
baseline methods like BPTT.
In conclusion, the proposed integer-only online learning algorithm presents an effective solution for efficiently training SNNs, enabling deployment on resource-constrained neuromorphic
hardware without sacrificing accuracy. Future research lines include achieving stability in deeper
networks, exploring more advanced neuron models and quantization techniques, and testing
sparsity methods to enhance efficiency and hardware applicability.

---

## Table of Contents
- [Models](#models)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Results](#results)
- [Citation](#citation)
- [Contact](#contact)

## Models

* Spiking Neural Network (SNN)
* Convolutional Spiking Neural Network (CSNN)
* Recurrent Spiking Neural Network (RSNN)

## Datasets

* MNIST
* Spiking Heidelberg Digits (SHD)

## Project Structure

- `model_snn/`: SNN model definitions
- `model_cnn/`: CSNN model definitions
- `model_rsnn/`: RSNN model definitions
- `grid_results_*`: Experiment results directories

## Results

MP 16-8: Mixed precision configuration with 16-bit shadow weights and 8-bit inference weights
FP32: Full precision floating point

### Accuracy on MNIST with SNN

| Inference weights |        Shadow weights         |     FP32      |
|-------------------|-------------------------------|---------------|
|                   | 8 bits         | 16 bits       |               |
| 4 bits            | 94.24 ± 0.27 % | 95.47 ± 0.15 %|               |
| 8 bits            | 96.89 ± 0.14 % | 97.55 ± 0.09 %| 97.33 ± 0.15 %|
| 16 bits           | -              | 97.43 ± 0.14 %|               |

### Accuracy on MNIST with CNN

| Inference weights |        Shadow weights         |     FP32      |
|-------------------|-------------------------------|---------------|
|                   | 8 bits         | 16 bits       |               |
| 4 bits            | 95.19 ± 0.84 % | 97.73 ± 0.19 %|               |
| 8 bits            | 97.16 ± 0.20 % | 98.10 ± 0.10 %| 97.85 ± 0.12 %|
| 16 bits           | -              | 97.88 ± 0.17 %|               |

### Accuracy on SHD with SNN and RSNN

| Model      | MP 16-4        | MP 16-8        | MP 16-12       | MP 16-16       | FP32            |
|------------|----------------|----------------|----------------|----------------|-----------------|
| SNN        | 49.85 ± 1.29 % | 50.10 ± 1.14 % | 62.06 ± 1.16 % | 61.92 ± 1.53 % | 55.27 ± 1.97 %  |
| SNN [21]   | -              | -              | -              | -              | 48.10 ± 1.60 %  |
| RSNN       | 57.62 ± 0.95 % | 64.63 ± 1.49 % | 70.50 ± 1.43 % | 67.75 ± 1.34 % | 71.64 ± 0.95 %  |
| RSNN [21]  | -              | -              | -              | -              | 71.40 ± 1.90 %  |

[21] Cramer, B., Stradmann, Y., Schemmel, J., Zenke, F.: The Heidelberg Spiking
Data Sets for the Systematic Evaluation of Spiking Neural Networks. IEEE
Transactions on Neural Networks and Learning Systems 33(7), 2744–2757 (Jul
2022). https://doi.org/10.1109/TNNLS.2020.3044364, https://ieeexplore.ieee.org/
document/9311226, conference Name: IEEE Transactions on Neural Networks and
Learning Systems

*See paper/thesis for full details and more results.*

## Citation

If you use this code or ideas in your research, please cite:
```bibtex
@mastersthesis{gomez2025integerSNN,
  title={Full Integer Arithmetic Online Training of Spiking Neural Network},
  author={Gomez, Ismael; Tang, Guangzhi},
  school={Maastricht University},
  year={2025}
}
```

## Contact

For questions or collaborations, contact: gomezgarrido.ismael@gmail.com

---

*Acknowledgments: Supervised by [Guangzhi Tang](https://www.maastrichtuniversity.nl/g-tang), Maastricht University.
Special thanks to Maastricht University.*
