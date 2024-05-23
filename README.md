# Pose similarity comparison using deep learning
This application helps to compare the similarity of poses of two different people in photos or videos

<details>
  <summary><b><u>Example</b> (clickable spoiler)</u></summary>
  
  * Image  
    ![](examples/images/img_comparison.png)
  
  * Video  
    ![](examples/images/video_gif_comparison.gif)
  
</details>

## Usage example
* **CMD/Bash**

  Call main script via `$ python main.py`, fill neccesary field and wait for output
* **As python class**

  Create instance of `VirtualCoach` from `./src/coach.py`, call function `.compare_poses`  
  [<b>Notebook with example</b>](./examples/usage_example.ipynb)

## Architecture
Simple visualization of pipeline:  
![](examples/images/arch.png)