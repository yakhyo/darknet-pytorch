## Implementation of DarkNet19, DarkNet53, CSPDarkNet53 on PyTorch

DarkNet19 and DarkNet53 are used as a feature extractor in [YOLO9000](https://arxiv.org/pdf/1612.08242.pdf)
, [YOLOv3](https://pjreddie.com/media/files/papers/YOLOv3.pdf) respectively.
<div align='center'>
  <img src='assets/darknet19.png' height="500px">
  <img src='assets/darknet53.png' height="500px">
</div>

##

## Description:

<table>
  <tr>
    <td></td>
    <td colspan="2" align="center">This Repo*</td>
    <td colspan="2" align="center">Official</td>
  </tr>
  <tr>
    <td>Model</td>
    <td>Acc@1</td>
    <td>Acc@5</td>
    <td>Acc@1</td>
    <td>Acc@5</td>
  </tr>
  <tr>
    <td>DarkNet19</td>
    <td><strong>70.5</strong></td>
    <td><strong>89.7</strong></td>
    <td>74.1</td>
    <td>91.8</td>
  </tr>
  <tr>
    <td>DarkNet53</td>
    <td><strong>75.6</strong></td>
    <td><strong>92.5</strong></td>
    <td>77.2</td>
    <td>93.8</td>
  </tr>
  <tr>
    <td>CSP-DarkNet53</td>
    <td><strong>74.1</strong></td>
    <td><strong>92.1</strong></td>
    <td>77.2</td>
    <td>93.6</td>
  </tr>
</table>

Weights of `DarkNet53` (105th epoch) and `DarkNet19` (50th epoch) are available
on [here](https://www.dropbox.com/sh/90it0q8tsclbpia/AAA0xcObKyndZ-r_Ia9vN1Xra?dl=0).
I am training the CSPDarkNet53 currently(the results shown above are the 65th epoch).

*Trained on ImageNet

- GPU: Tesla V100
- Input size: 3x224x224

Dataset structure:

```
├── IMAGENET 
    ├── train
    ├── val
```

## Train:

```
 git clone https://github.com/yakhyo/DarkNet.git
 cd DarkNet2
 python main.py ../IMAGENET --batch-size 512 --workers 8
```

**Note**

Modify [this line](https://github.com/yakhyo/DarkNet/blob/bf1d0c50935d71fa3918fa65060f73c047733acf/main.py#L52) to choose the network to start the training:

```python
# darknet19
model = darknet19(num_classes=1000, init_weight=True)

# darknet53
model = darknet53(num_classes=1000, init_weight=True)

# cspdarknet53
model = cspdarknet53(num_classes=1000, init_weight=True)
```

## Continue the training:
```cmd
python main.py ../../Dataset/IMAGENET --batch-size 512 --workers 8 --resume darknet53.pth.tar
```
