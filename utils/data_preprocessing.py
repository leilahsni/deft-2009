import pandas as pd
import matplotlib.pyplot as plt
from utils import SEED

def distrib_visualization(df, filename: str):

	fig = plt.figure(figsize=(10,10))
	df.groupby('parti').doc.count().plot.bar(ylim=0)
	plt.savefig(filename)

def normalize_data_distrib(df, val, b=0):

	n = len(df[df.parti == 'ELDR'])

	df_without_n = df.loc[df.parti != val]
	df_n = df.loc[df.parti == val].sample(n=n+b, random_state=SEED)

	return pd.concat([df_without_n, df_n])