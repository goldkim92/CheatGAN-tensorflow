# CheatGAN-tensorflow

*concat experiment with AEDG*
```
python main.py --phase=train --gpu_number=0 --loss_type=GAN --assets=no_connect/zdim2_ae10 --z_dim=2 --lambda_ae=10
```

*concat experiment with Vanilla GAN (using DCGAN)*
```
python main.py --phase=train --batch_size=128 --lr=0.0002 --beta1=0.5 --gpu_number=2 --loss_type=GAN --data=celebA --assets=celebA/zdim100/vanilla --z_dim=100
```
