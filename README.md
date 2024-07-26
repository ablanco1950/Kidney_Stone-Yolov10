# Kidney_Stone-Yolov10
From dataset https://www.kaggle.com/datasets/safurahajiheidari/kidney-stone-images/data (from Roboflow) a model is obtained, based on yolov10, with that custom dataset, to indicate Kidney_Stone in x -rays.

=== Installation: ======== Download all project datasets to a folder on disk.

Install yolov10 (if not yet installed) following the instructions given at:

https://blog.roboflow.com/yolov10-how-to-train/

which may be reduced to !pip install -q git+https://github. com/THU-MIG/yolov10.git 

And download from https://github.com/THU-MIG/yolov10/releases the yolov10n.pt model. In case this operation causes problems, this file is attached with the rest of the project files.

Unzip the test1.zip folder Some zip decompressors duplicate the name of the folder to be decompressed; a folder that contains another folder with the same name, should only contain one. In these cases it will be enough to cut the innermost folder and copy it to the project folder.

If there are problems when executing the following steps and if you have already installed Yolov10 previously, it would be necessary to upgrade yolov10 and the lap version:

inside .conda in the scripts directory of the user environment

python pip-script.py install --no -cache-dir "lapx>=0.5.2"

upgrade ultralytics:

python pip-script.py install --upgrade ultralytics

===
Test:

It is executed:

EvaluateKidneyStoneYolov10.py

The x-rays are presented on the screen with a red box indicating the prediction and in green showing the true location of the stone or how it has been labeled.

Messages appear on the console indicating how many stones have been detected and how many actually exist.

Generally the detected box coincides with the true one, so it results in a a little confusing.

===

Training 

The project comes with an model: last39epoch.pt, to obtain this model, the following has been executed:

Download de file https://www.kaggle.com/datasets/safurahajiheidari/kidney-stone-images/data ( is from roboflow)

After downloading the dataset a folder is created with subfolders: train, valid and test.

Put this 3 folders in the folder directory of project  and execute

TrainKidneyStone.py 

This program has been adapted from https://medium.com/@huzeyfebicakci/custom-dataset-training-with-yolov10-a-deep-dive-into-the- latest-evolution-in-real-time-object-ab8c62c6af85 It assumes that the project is located in the folder "C:/Kidney_Stone-Yolov10", otherwise the assignment must be changed by modifying line 20 The parameter multi_scale has been changed to true .

also uses the .yaml file: data.yaml. In data.yaml the absolute addresses of the project appear assuming that it has been installed on disk C:, if it has another location these absolute addresses will have to be changed.

LOG_train_kidney_stone.docx is attached, in which it can be observed that in epoch 39 a mAP50 of 0.718 and a mAP50-95 of 0.271 are obtained

as the training is carried out, with each epoch, in the runs\\train\\exp directory \\weights\\ of the project folder, the models best.pt (the supposedly best one obtained) and last.pt, the last one obtained, are saved.

References:

https://www.kaggle.com/datasets/safurahajiheidari/kidney-stone-images/data (roboflow dataset

https://medium.com/@huzeyfebicakci/custom-dataset-training-with-yolov10-a-deep-dive-into-the-latest-evolution-in-real-time-object-ab8c62c6af85

https://medium.com/@girishajmera/fine-tuning-yolov10-for-custom-object-detection-7b12093691c8

