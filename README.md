# DS6050-Toxic-Plant-Classification
The repository contains materials related to Group 13's Toxic Plant Classification Project. 

The repository includes the folders/files: 
- Milestone 1 
- Milestone 2 and Final Project Files
- requirements.txt

## Structure of MilestoneII and Final Project Files
Expected layout:
/ (project root)

main.py — entry point for CNN training/evaluation

data/ - contains data processing code

train/ — contains CNN-training code (train_cnn.py, train_rf.py)

eval/ — contains evaluation code (evaluation_cnn.py, evaluation_rf.py)

features/ — feature extraction code for RF pipeline (rf_features.py)

input/ — root directory containing image data and metadata (images + metadata CSV)

README.txt — this file

## Milestone 1: Literature Review I
Contains a PDF file with a literature review and initial project proposal.

## Literature Review II & Final Project
Contains a PDF file with an expanded literature review, initial toxic plant classification model training and evaluation results, and outlined next steps. The subfolder `\input` contains the plant image data for toxic and non-toxic plants that were used for training, validation, and testing. 

## Final Project - Run and Evaluate Models
You should install necessary Python packages before running that are contained within the requirements.txt.

Ensure your image data and metadata CSV (with image paths & labels) under the expected directory structure (input folder) so that data-loading logic can find them correctly.

Usage & command-line options (CNN pipeline)
Run from project root (where main.py lives)

## Available options / flags:

--mode {train, eval, all}
Choose operation mode:
* train — trains a specified CNN model (and optionally evaluates afterwards)
* eval — loads a saved model (with --model_path) and evaluates on test data
* all — loops through all model variants (b0, b0_cbam, b2_cbam), training (and optionally evaluating) each

`--model {b0, b0_cbam, b2_cbam}`
CNN variant to use (required for mode = train, or used in mode = all)

`--pretrained`
If provided, use pretrained ImageNet weights for the backbone

`--dropout <float>`
Dropout probability for classifier head (default 0.2)

`--epochs <int>`
Number of training epochs (default 10)

`--lr <float>`
Learning rate for optimizer (default 1e-4)

`--strong_aug`
If provided, use stronger data augmentation during training

`--evaluate_after_train`
When training (mode train or all), if provided, run evaluation on test data immediately after training

`--model_path <path>`
Used when mode = eval: path to saved model file to load and evaluate

## Example usage:

Train a CNN model (no evaluation):

`python main.py --mode train --model b0_cbam --pretrained --strong_aug --epochs 10`


Train + Evaluate a CNN model:

`python main.py --mode train --model b0_cbam --pretrained --strong_aug --epochs 10 --evaluate_after_train`


Evaluate a saved CNN model:

`python main.py --mode eval --model_path path/to/saved_model.pth`


Train (and optionally evaluate) all CNN variants:

`python main.py --mode all --pretrained --strong_aug --epochs 10 --evaluate_after_train`


RF pipeline usage (feature-based):

`python train_rf.py --meta_csv path/to/metadata.csv --image_root path/to/images --output_model rf_model.joblib`

When using mode “all,” running through multiple model variants will take longer — you are retraining/evaluating several models sequentially.

# Additional Train CNN Examples
`python main.py --model b0_cbam --pretrained --strong_aug --epochs 10`

`python main.py --mode train --model b0_cbam --pretrained --strong_aug --epochs 10 --lr 0.0001`

`python main.py --mode train --model b0_cbam --pretrained --strong_aug --epochs 10 --lr 0.0001 --evaluate_after_train`

`python main.py --mode eval --model_path path/to/saved_model.pth (model that has already been run for evauation)`

`python main.py --mode all --pretrained --strong_aug --epochs 10 --lr 0.0001 --evaluate_after_train`

# Additional Train RF Examples
`python train_rf.py --meta_csv data.csv --image_root ./images`



