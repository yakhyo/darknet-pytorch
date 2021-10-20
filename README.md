## Implementation of DarkNet on PyTorch

DarkNet19 and DarkNet53 are used as a feature extractor in [YOLO9000](https://arxiv.org/pdf/1612.08242.pdf), [YOLOv3](https://pjreddie.com/media/files/papers/YOLOv3.pdf) respectively.
<div align='center'>
  <img src='assets/darknet19.png' height="550px">
  <img src='assets/darknet53.png' height="550px">
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
    <td>...</td>
    <td>...</td>
    <td>74.1</td>
    <td>91.8</td>
  </tr>
  <tr>
    <td>DarkNet53</td>
    <td><strong>72.2</strong></td>
    <td><strong>90.8</strong></td>
    <td>77.2</td>
    <td>93.8</td>
  </tr>
</table>
*Trained on ImageNet (50 epochs)

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
 cd DarkNet
 python main.py ../IMAGENET --batch-size 512 --workers 8
```
