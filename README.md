Yoga Classifier
==============================

Yoga has become increasingly popular over the years, providing many health benefits. In our project, we aim to create a Multilayer Perceptron model that can identify yoga poses. We use BlazePose 3D to identify the keypoints and calculate the normalized distances. Our testing accuracy is 100%, a very good result. Our recommendation for future work is to use a dataset with a higher variety of poses and return feedback about the performed exercises.

**The report is in the report folder.**

## How tu run the code

### Create an environment and activate it:
 ```
conda create --name ci-yoga python=3.9
conda activate ci-yoga
pip install -r requirements.txt
 ```

### The main code is in the `yoga` folder:

- To process data `yoga/process_data`:
    - make_dataset.py: to transform images into datasets of key points
- Models `yoga/models`:
    - build_features.py : to transform the pose keypoints dataset into final_dataset.csv
    - train_model.py: train three classifiers and save performance information in test_sats.csv
    - analysis.py: useful plots to compare the models