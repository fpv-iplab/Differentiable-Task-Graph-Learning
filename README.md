# Differentiable Task Graph Learning: Procedural Activity Representation and Online Mistake Detection from Egocentric Videos

[Differentiable Task Graph Learning: Procedural Activity Representation and Online Mistake Detection from Egocentric Videos](https://arxiv.org/abs/2406.01486)

[Luigi Seminara](https://seminaraluigi.altervista.org/), [Giovanni Maria Farinella](https://www.dmi.unict.it/farinella/), [Antonino Furnari](https://antoninofurnari.github.io/)

NeurIPS (spotlight), 2024

[arXiv pre-print](https://arxiv.org/pdf/2406.01486v1)

## ✍️ Catalog
🚧 WORK IN PROGRESS:
- [x] Baselines
- [x] Direct Optimization (DO) Model
- [ ] Task Graph Transformer (TGT) Model
- [ ] Online Mistake Detection

---
- [Environment configuration](#environment-configuration)
- [Data](#data)
- [Training](#training)
- [Qualitative results](#qualitative-results)
- [Get DO results of Table 1 of the paper](#get-do-results-of-table-1-of-the-paper)
- [Citation](#citation)
- [Authors](#authors)
---

<p align="center">
  <img src="./assets/tgml.png" width="70%" height="auto">
</p>

## Environment configuration

The code was tested with Python 3.9. Run the following commands to configurate a new conda environment:

```shell
conda create -n tgml python=3.9
conda activate tgml
python -m pip install -e ./lib
conda install -c conda-forge pygraphviz
```

The specified versions of PyTorch and its associated libraries are recommended for optimal compatibility and performance:

- **PyTorch:** 2.0.1
- **Torchvision:** 0.15.2
- **Torchaudio:** 2.0.2
- **PyTorch with CUDA:** Version 11.7

These packages can be installed using the following command:
```bash
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

While these versions are recommended, newer versions of these libraries may also be compatible with the project. If you choose to use alternative versions, please ensure they do not introduce any compatibility issues. 



## Data
In the **./data** directory, you will find the CaptainCook4D data that we have defined for our task. This data is provided in compliance with the license defined by the original authors. Our split differs from those defined by the original authors of the paper, as we have only included annotations that do not contain errors. For more information about the original dataset, please visit the official [CaptainCook4D repository](https://github.com/CaptainCook4D/).



## Qualitative results
The figure reports the generated task graphs of the procedure called "Dressed Up Meatballs". On the left there is the ground truth task graph, while on the right the generated using the Direct Optimization model. These graphs must be interpreted from the bottom up, reflecting the bottom-up nature of dependency edges.

Ground Truth             |  Generated 
:-------------------------:|:-------------------------:
![First Image](./assets/task_graph_ground_truth.png) | ![Second Image](./assets/task_graph_generated.png)



## Citation
If you use the code/models hosted in this repository, please cite the following paper:

```text
@misc{seminara2024differentiable,
      title={Differentiable Task Graph Learning: Procedural Activity Representation and Online Mistake Detection from Egocentric Videos}, 
      author={Luigi Seminara and Giovanni Maria Farinella and Antonino Furnari},
      year={2024},
      eprint={2406.01486},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
Please, refer to the [paper](https://arxiv.org/pdf/2406.01486v1) for more technical details.


## Authors

![Authors](./assets/authors/authors.png)
