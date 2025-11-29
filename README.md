# DS6050-Toxic-Plant-Classification
The repository contains materials related to Group 13's Toxic Plant Classification Project. 

The repository includes the folders/files: 
- Milestone 1 
- Milestone 2
- requirements.txt

## Milestone 1: Literature Review I
Contains a PDF file with a literature review and initial project proposal.

## Milestone 2: Literature Review II
Contains a PDF file with an expanded literature review, initial toxic plant classification model training and evaluation results, and outlined next steps. The subfolder `\input` contains the plant image data for toxic and non-toxic plants that were used for training, validation, and testing. 

## Final Project - Run and Evaluate Models
# Train CNN
python main.py --model b0_cbam --pretrained --strong_aug --epochs 10
# Train RF
python train_rf.py --meta_csv data.csv --image_root ./images
