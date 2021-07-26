# Attention-guided Temporally Coherent Video Object Matting Trimap Generation Module

Here we provide the inference (online-finetuning included) code and pretrained weight for our trimap generation module in our paper **Attention-guided Temporally Coherent Video Object Matting**.

**The code, the trained model are for academic and non-commercial use only.**

This module is based on STM by Oh et al., code is from [here](https://github.com/seoungwugoh/STM).

## Requirements

```
Python=3.8
Pytorch=1.6.0
numpy
opencv-python
imgaug
tqdm
yacs
```

## Usage

* Download the pretrained weight [here](https://1drv.ms/u/s!AuG441T6ysq5hWA0GbH4A0tDfgWd?e=N4gD8i).

* Prepare the video clip data, make sure the annotated trimap and the RGB image correspond each other. For example, say the video clip has 100 frames and the first, the last, and the 50th frame have their corresponding keyframe trimaps. The video clip folder should have the structure of the following:

  ```
  |---workspace
      |---video_clip
          |---00000_rgb.png
          |---00000_trimap.png
          |---00001_rgb.png
          |---00002_rgb.png
          |---...
          |---00049_rgb.png
          |---00049_trimap.png
          |---...
          |---00097_rgb.png
          |---00098_rgb.png
          |---00099_rgb.png
          |---00099_trimap.png
  ```

* Create a training configuration yaml for the video clip. Please refer to `template.yaml` for detail.

* Online-finetuning:

  ```bash
  python online_finetune_test.py --cfg /path/to/config.yaml
  ```

* Inference:

  ```bash
  python test_mf_2.py --cfg /path/to/config.yaml --load /path/to/finedtuned_weight.pth --save /path/to/results
  ```

  * If you ran out of memory during the inference, there are two parameters you can tune.
    * `mem_every`: STM 'memorizes' a new feature map every 5 frames by default. Try increasing this number.
    * `max_memory`: The maximum number of feature maps inside STM's memory pool. By setting this parameter, STM will 'forget' the oldest feature map (except the ground-truth frame) in the memory pool to cap the GPU memory usage. By default it is `None`, the memory pool will grow indefinitely.

* Ensemble:

  ```bash
  python ensemble.py --predroot /path/to/results --dataroot /path/to/workspace [video_clip_folder1] [video_clip_folder2] ...
  ```

  * By default the script will process all folders under `/path/to/results`. You can also specify the folder name that you want to process in the command-line.
  * The final result is under `/path/to/results/video_clip_folder/ensemble`.


## Contact

If you have any questions, please feel free to contact `yunkezhang@zju.edu.cn`.
