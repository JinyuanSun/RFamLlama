# RFamLlama
A pretrained LM for RNA



## Train the model
```bash
wget http://103.79.77.89/rfam_f90_train.csv.gz
wget http://103.79.77.89/rfam_f90_test.csv.gz

# train a small model

python train.py --model_size small --dataset rfam_f90 > rfam_f90_small.log 
```

## Zero-shot fitness prediction
**The best**, *The second*
| Method         | tRNA (Li et al) | glmS ribozyme (Andreasson et al) | glmS ribozyme (Sumi et al) | drz-agam-2-1 ribozyme, (Kobori et al) | Twister ribozyme P1 (Kobori et al) | Average |
|----------------|-----------------|----------------------------------|---------------------------|---------------------------------------|------------------------------------|---------|
| RfamGen        | **0.556**           | 0.546                            | 0.371                     | *0.035*                                 | *0.425*                              | **0.387**   |
| EVMutation     | 0.493           | **0.657**                            | 0.321                     | -0.121                                | **0.548**                              | 0.380   |
| RFamLlama-small  | *0.503*           | 0.475                            | 0.397                     | 0.049                                 | 0.391                              | 0.363   |
| RFamLlama-base   | 0.460           | 0.518                            | **0.443**                     | 0.016                                 | 0.421                              | 0.372   |
| RFamLlama-large  | 0.427           | *0.584*                            | *0.407*                     | **0.077**                                 | 0.269                              | 0.353   |
