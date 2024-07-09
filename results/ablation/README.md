## Ablation on Data: Data Quality Matters
Here, we provide two simple implementation notebooks corresponding to the data quality ablation experiments in our paper (Figure 6). By altering the brightness of the input images, we observe the impact of data changes on pruning methods. The key implementations are as follows:
```python
# utils/data_utils.py
# Line 74
    transforms.ColorJitter(brightness=(config.brightness, config.brightness)),
```
The above statement altered the brightness of input images before data augmentation, allowing us to observe the varying degrees of layer collapse in the pruned VGG11 model.