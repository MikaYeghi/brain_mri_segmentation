# brain_mri_segmentation
Brain MRI segmentation task from Kaggle.

To get the data set up, follow these steps:
1. Download the [data set](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) from Kaggle
2. Place the raw data set in the "raw_data/" directory
3. Create a "data/" directory by typing ``mkdir data`` in the terminal in the root directory
4. Run ``python sort_data.py`` to randomly shuffle the data and generate train, validation and test sets

After the data has been placed in the relevant "data/train/", "data/val/" and "data/test/" folders, run training by typing:

```
python train.py
```

After training a model, run inference on the test set by typing:

```
python inference.py
```

All predictions and their relevant masks will be saved to the "predictions/" directory. Running the inference script will also print the mean IoU score of the model predictions on the test set.
