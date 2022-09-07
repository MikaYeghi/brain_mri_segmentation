# brain_mri_segmentation
Brain MRI segmentation task from Kaggle.

To get the data set up, follow these steps:
1. Download the [data set](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) from Kaggle
2. Place the raw data set in the "raw_data/" directory
3. Create a "data/" directory by typing ``mkdir data`` in the terminal in the root directory
4. Run ``python sort_data.py`` to randomly shuffle the data and generate train, validation and test sets
