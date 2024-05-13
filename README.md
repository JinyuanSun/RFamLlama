# RFamLlama
A pretrained LM for RNA



## Train the model
```bash
wget http://103.79.77.89/rfam_f90_train.csv.gz
wget http://103.79.77.89/rfam_f90_test.csv.gz

# train a small model

python train.py --model_size small --dataset rfam_f90 > rfam_f90_small.log 
```
