# Fork of Auto-Encoder trained on ImageNet

First run:

```
ffmpeg -i custom_game.mp4 -vf scale=256:256 -r 6 tmp/extr5/image-%3d.png
```

This will create the folder of images sampled at 6Hz.

Then create the list for the dataset with:

```
python tools/generate_list.py --name {name your dataset such as caltech256} --path {path to your dataset}
```


Train the autoencoder with (we provide the checkpoint on demand):
```
python train.py --arch resnet50 --train_list list/NAME_list.txt --batch-size 6 --workers 1 --start-epoch 10 --epochs 11 --pth-save-fold outputs 
```

Finally evaluate the results with:
```
python run_autoencoder.py
```

This will produce an arr.csv which is used in the Dota_2_Highlight project.

