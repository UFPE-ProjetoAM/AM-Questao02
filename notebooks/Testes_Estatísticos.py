import numpy as np
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
import pandas as pd


# Carrega o CSV em um DataFrame

df = pd.read_csv('mean_by_repetition_and_model.csv')
print(df)
Metrica = "Precision"
LR_accuracy = df[df['Model'] == 'LR'][Metrica].tolist()
GB_accuracy = df[df['Model'] == 'GB'][Metrica].tolist()
PW_accuracy = df[df['Model'] == 'PW'][Metrica].tolist()
KNN_accuracy = df[df['Model'] == 'KNN'][Metrica].tolist()

print(LR_accuracy)
print(GB_accuracy)
print(PW_accuracy)
print(KNN_accuracy)

# Teste de Friedman

res = friedmanchisquare(LR_accuracy, GB_accuracy, PW_accuracy, KNN_accuracy)


estatistica = res.statistic
valor_p = res.pvalue

print(f"Estatística de Teste de Friedman: {estatistica}")
print(f"Valor-p: {valor_p}")

#Teste de Nemenyi

data = np.array([LR_accuracy, GB_accuracy, PW_accuracy, KNN_accuracy])

sp.posthoc_nemenyi_friedman(data.T)
print(f"A comparação dos pares através do teste de Nemenyi: \n {sp.posthoc_nemenyi_friedman(data.T)}")

