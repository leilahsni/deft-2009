# DEFT 2009 - Task n°3
### Introduction

This repository contains the scripts and datasets used to train a supervised classification model for task n°3 of the DEFT (DEfi Fouille de Texte) 2009 edition. The task consists of automatically detecting the political party of a speaker based on datasets given by the DEFT event holders. The datasets are comprised of speeches made by five different political parties at the European Parliament : PPE-DE, Verts-ALE, PSE, GUE-NGL and ELDR. 

The datasets and in-depth description of the task (in French) can be found [here](https://deft.lisn.upsaclay.fr/2009/index.php?id=2&lang=fr).


### Run the script

In your virtual environement:

1.

> pip install -r requirements.txt

2.

> python classification_model.py

Two TSV files will be automatically created in data/tsv : deft-2009-test.tsv and deft-2009-train.tsv

On the first run, the model.sav file will be created automatically in the model directory. If a model.sav file already exists, you will be asked wether you want to overwrite the previous model or not : type *y* for yes, *n* for no.

The results (confusion matrix, classification report, model predictions, etc.) can all be found in the eval directory.