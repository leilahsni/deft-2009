1. Pour installer toutes les librairies nécessaires, dans un environnement virtuel : pip install -r requirements.txt
2. Lancer le script classification_model.py
3. Si les fichiers data/deft-2009-test.tsv et data/deft-2009-train.tsv n'existent pas, ils sont crées automatiquement.
4. Si un modèle existe déjà dans le dossier model, le script vous demandera si vous souhaitez ré-entraîner le modèle. Taper y pour oui, n pour non. Si le fichier model.sav n'existe pas, il est créer automatiquement.
5. Vérifier les résultats (matrice de confusion, classification report, prédictions du modèle, etc.) dans le dossier eval

NB : les scripts pour le preprocessing et le formatage des données se trouvent dans utils