## Implementation of DarkNet19, DarkNet53, CSPDarkNet53 on PyTorch

## Contents:
 
1. [**DarkNet19**](https://arxiv.org/pdf/1612.08242.pdf) - used as a feature extractor in **YOLO900**.
2. [**DarkNet53**](https://pjreddie.com/media/files/papers/YOLOv3.pdf) - used as a feature extractor in **YOLOv3**.
3. **CSPDarkNet53** - Implementation of [**Cross Stage Partial Networks**](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w28/Wang_CSPNet_A_New_Backbone_That_Can_Enhance_Learning_Capability_of_CVPRW_2020_paper.pdf) in **DarkNet53**.
4. **DarkNet53-Elastic** - Implementation of [**ELASTIC**](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_ELASTIC_Improving_CNNs_With_Dynamic_Scaling_Policies_CVPR_2019_paper.pdf) with **DarkNet53**. 
5. **CSPDarkNet53-Elastic** - Implementation of **CSP** and **ELASTIC** in **DarkNet53**._??_

Architecture of [**DarkNet19**](https://arxiv.org/pdf/1612.08242.pdf) and [**DarkNet53**](https://pjreddie.com/media/files/papers/YOLOv3.pdf):

<div align='center'>
  <img src='assets/darknet19.png' height="500px">
  <img src='assets/darknet53.png' height="500px">
</div>

##

## Description:

### Results:

<table>
  <tr>
    <td></td>
    <td colspan="3" align="center">This Repo.</td>
    <td colspan="2" align="center">Official</td>
  </tr>
  <tr>
    <td>Model</td>
    <td>Acc@1</td>
    <td>Acc@5</td>
    <td>Params</td>
    <td>Acc@1</td>
    <td>Acc@5</td>
  </tr>
  <tr>
    <td>DarkNet19</td>
    <td><strong>70.5</strong></td>
    <td><strong>89.7</strong></td>
    <td>21M</td>
    <td>74.1</td>
    <td>91.8</td>
  </tr>
  <tr>
    <td>DarkNet53</td>
    <td><strong>75.6</strong></td>
    <td><strong>92.5</strong></td>
    <td>41M</td>
    <td>77.2</td>
    <td>93.8</td>
  </tr>
  <tr>
    <td>CSP-DarkNet53</td>
    <td><strong>74.3</strong></td>
    <td><strong>92.2</strong></td>
    <td>19M</td>
    <td>77.2</td>
    <td>93.6</td>
  </tr>
<tr>
    <td>DarkNet53-Elastic</td>
    <td><strong>70.8</strong></td>
    <td><strong>90.2</strong></td>
    <td>24M</td>
    <td>...</td>
    <td>...</td>
  </tr>
<tr>
    <td>CSPDarkNet53-Elastic</td>
    <td><strong>...</strong></td>
    <td><strong>...</strong></td>
    <td><strong>...</strong></td>
    <td>76.1</td>
    <td>93.3</td>
  </tr>
</table>

Weights of `DarkNet53` (105th epoch), `DarkNet19` (50th epoch), `CSPDarkNet53` (80th epoch) and `DarkNet53 ELASTIC` (57th epoch) are available
on [here](https://www.dropbox.com/sh/90it0q8tsclbpia/AAA0xcObKyndZ-r_Ia9vN1Xra?dl=0).


Trained on ImageNet

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
from nets.nn import darknet19, darknet53, darknet53e, cspdarknet53

# darknet19
model = darknet19(num_classes=1000, init_weight=True)

# darknet53
model = darknet53(num_classes=1000, init_weight=True)

# darknet53 elastic
model = darknet53e(num_classes=1000, init_weight=True)

# cspdarknet53
model = cspdarknet53(num_classes=1000, init_weight=True)
```

## Continue the training:
```cmd
python main.py ../../Dataset/IMAGENET --batch-size 512 --workers 8 --resume darknet53.pth.tar
```
