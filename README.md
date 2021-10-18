## Implementation of DarkNet on PyTorch

DarkNet19 and DarkNet53 are used as a feature extractor in [YOLO9000](https://arxiv.org/pdf/1612.08242.pdf), [YOLOv3](https://pjreddie.com/media/files/papers/YOLOv3.pdf) respectively.

*after 30 epochs on ImageNet (trained around 2 days on 3 Tesla V100:

| Model | Acc@1 | Acc@5 | 
|-------|------|--------|
| Darknet53 | 70.59 | 90.01 |
| Darknet19 | Dice | 81.65 |

<div align='center'>
  <img src='assets/darknet19.png' height="400px">
  <img src='assets/darknet53.png' height="400px">
</div>



## Description:


Run:
```
   git clone https://github.com/yakhyo/DarkNet.git
   cd DarkNet
   python main.py --batch-size 64 [data folder]

```

## Content:

- Data
- Training
- Evaluation and Inference

## Data:

- Download the training files from [here]() and place them as shown below:
