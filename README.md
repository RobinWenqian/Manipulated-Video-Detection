<a href="http://fvcproductions.com"><img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gge32669o5j30su0nkqom.jpg" title="Deepfake Detection" alt="FVCproductions"></a>

<!-- [![FVCproductions](https://tva1.sinaimg.cn/large/007S8ZIlgy1gge32669o5j30su0nkqom.jpg) -->

# Manipulated Video Detection

> Manipulated Video Detection with Pytorch

> Video Dataloader ready-to-use


## Environment Needed

- Python 3.7
- Pytorch >= 1.00
- torchvision 0.4.0
- opencv 3.4.2
- etc.

## Method Introduction
- MesoNet-Inception

<a href="http://fvcproductions.com"><img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ggb4z53xswj30ic0ga0ub.jpg" title="Deepfake Detection" alt="FVCproductions"></a>

- MesoNet

<a href="http://fvcproductions.com"><img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gge5jgelv7j30hu0mmjty.jpg" title="Deepfake Detection" alt="FVCproductions"></a>

- Other Structure (Future Work)

## Installation

- All the `code` required to get started
- pip install -r requirements.txt

### Clone

- Clone this repo to your local machine using `https://github.com/RobinWenqian/Manipulated-Video-Detection.git`



## Data Preparation

- This project is for Celeb-v2 dataset (http://www.cs.albany.edu/~lsw/celeb-deepfakeforensics.html) and you can modify the video list generator and dataloader as you need for your customized datasets.

## Usage

- run `video_list_gen.py` to generate training data list and test data list

- run `txt_pickle_convert.py` to transfer data list from .txt to .pkl. This process is for Video_Dataloader to read.

- write a `run.sh` to customize your own hyper-parameters (optional) or directly run `train_eval.py` 

## Contributing

> To get started...

### Step 1

- **Option 1**
    - 🍴 Fork this repo!

- **Option 2**
    - 👯 Clone this repo to your local machine using `https://github.com/joanaz/HireDot2.git`

### Step 2

- **HACK AWAY!** 🔨🔨🔨

### Step 3

- 🔃 Create a new pull request using <a href="https://github.com/RobinWenqian/Manipulated-Video-Detection/pulls" target="_blank">`https://github.com/RobinWenqian/Manipulated-Video-Detection/pulls`</a>.


## Contributer

Wenqian (Bradley) He - Graduated from the University of Edinburgh and Shanghai Jiao Tong University

## Support

Reach out to me at one of the following places!

- Personal E-mail: vincent_wq@outlook.com
- Edu E-mail: s1946842@ed.ac.uk



## License

- **[MIT license](http://opensource.org/licenses/mit-license.php)**
- Copyright 2020 
