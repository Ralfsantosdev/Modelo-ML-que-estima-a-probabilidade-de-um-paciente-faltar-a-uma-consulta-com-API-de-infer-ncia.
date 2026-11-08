# Comparação (conjunto de TESTE, limiar da validação)

Melhor por ROC-AUC na validação: **logistic_regression**

| modelo              |   limiar |   test_precision |   test_recall |   test_f1 |   test_roc_auc |   test_pr_auc |
|:--------------------|---------:|-----------------:|--------------:|----------:|---------------:|--------------:|
| logistic_regression |   0.4463 |           0.3315 |        0.6798 |    0.4457 |         0.6304 |        0.3875 |
| random_forest       |   0.4431 |           0.3258 |        0.6459 |    0.4331 |         0.6254 |        0.3756 |
| xgboost             |   0.4565 |           0.3237 |        0.6358 |    0.429  |         0.6206 |        0.378  |
