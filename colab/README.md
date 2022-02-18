## Обучение GAN моделей на Google Colab

Ниже представлены результаты обучения моделей для этапов `TOTAL_ITERS == 1500`, `TOTAL_ITERS == 1800` итераций при использовании GPU.  

### Результаты при `TOTAL_ITERS == 1500`
```
gan_training = mixup_LSGAN
use_PL = True
PL_before_activ = True
use_mask_hinge_loss = False
m_mask = 0.0
lr_factor = 0.1
use_cyclic_loss = False
----------
[iter 1500] Loss_DA: 0.514674 Loss_DB: 0.573655 Loss_GA: 2.045384 Loss_GB: 1.867255 time: 658.915660
----------
Детали потерь генератора:
\[Adversarial loss]
GA: 0.0715 GB: 0.0715
[Reconstruction loss]
GA: 0.8642 GB: 0.7406
[Edge loss]
GA: 0.5377 GB: 0.4928
[Perceptual loss]
GA: 0.2616 GB: 0.2521
----------
```

**Преобразованные (замаскированные) результаты для 1500 итераций:**
![Preprocessed for 1500 iterations](../doc/images/train_preprocessed_1500.png)

**Маски**:
![Masked for 1500 iterations](../doc/images/train_masked_1500.png)

**Результаты реконструкции**:
![Masked for 1500 iterations](../doc/images/train_reconstruction_1500.png)

### Результаты при `TOTAL_ITERS == 1800`

```
gan_training = mixup_LSGAN
use_PL = True
PL_before_activ = True
use_mask_hinge_loss = False
m_mask = 0.0
lr_factor = 0.1
use_cyclic_loss = False
----------
[iter 1800] Loss_DA: 0.490654 Loss_DB: 0.492488 Loss_GA: 1.937289 Loss_GB: 1.748453 time: 1107.671602
----------
Детали потерь генератора:
[Adversarial loss]
GA: 0.0620 GB: 0.0704
[Reconstruction loss]
GA: 0.8007 GB: 0.6668
[Edge loss]
GA: 0.5323 GB: 0.4849
[Perceptual loss]
GA: 0.2487 GB: 0.2325
----------
```

**Преобразованные (замаскированные) результаты для 1800 итераций:**
![Preprocessed for 1800 iterations](../doc/images/train_preprocessed_1800.png)

**Маски**:
![Masked for 1800 iterations](../doc/images/train_masked_1800.png)

**Результаты реконструкции**:
![Masked for 1800 iterations](../doc/images/train_reconstruction_1800.png)