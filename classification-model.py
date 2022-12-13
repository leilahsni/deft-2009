import pickle
import os

import seaborn as sns
import sklearn.metrics as metrics

from utils.data_formating import *
from utils.data_preprocessing import *

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

class ClassificationModel():
	''' instance of classification model '''

	def __init__(self, model_filename='model/model.sav', vectorizer=TfidfVectorizer(encoding='utf-8', lowercase=True, token_pattern=r"\b\w+?\b(?:'|(?:-\w+?\b)*)?", min_df=5, max_df=.3, norm='l2', ngram_range=(2,2)), model=LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')):

		self.vectorizer = vectorizer
		self.model_filename = model_filename
		self.model = model

	def fit_X_y_train(self, train):
		''' function vectorize train data and train model '''

		X_train,y_train = train.doc.values.astype(str), train.parti.values.astype(str)
		X_train = self.vectorizer.fit_transform(X_train)

		return X_train, y_train

	def train_model(self, X, y):
		''' train model '''

		if exists(self.model_filename):
			userInput = input(f'Found a {self.model_filename} file in directory. Want to overwrite previous model ? [y/n] : ')
			while userInput.lower() not in ['y', 'n']:
				userInput = input(f'Invalid input : {userInput}. Want to overwrite previous model ? [y/n] : ')

			if userInput.lower() == 'y':
				print('Training model...')
				self.model.fit(X_train,y_train)
				print('Saving model...')
				pickle.dump(self.model, open(self.model_filename, 'wb'))
		else:
			print('No previous model found in directory. Training model...')
			self.model.fit(X_train,y_train)
			print('Saving model...')
			pickle.dump(self.model, open(self.model_filename, 'wb'))

	def fit_X_y_test(self, test):
		''' vectorize test set '''

		X_test,y_test = test.doc.values.astype(str), test.parti.values.astype(str)
		X_test = self.vectorizer.transform(X_test)

		return X_test, y_test

	def evaluate(self, model, X, y):
		''' get model performance on test set '''

		return round(model.score(X_test,y_test)*100)

	def get_model_preds(self, model, X):
		''' get model predictions on test set '''

		return model.predict(X_test)

	def predict_new_input(self, model, *args):
		''' get prediction of new text input from model '''

		preds = []

		for text in args:
			text = self.vectorizer.transform([text])
			pred = model.predict(text)
			preds.append(''.join(pred))

		return preds

if __name__ == '__main__':

	# creates TSV files from original XML files if these files don't exist
	fetch_data(f_type='test', xml_infile=IN_TEST, outfile=OUT_TEST)
	fetch_data(f_type='train', xml_infile=IN_TRAIN, outfile=OUT_TRAIN)

	cm = ClassificationModel() # create instance of classification model
	eval_outdir = './eval/'

	# keeping 25 to 50% of datasets for memory reasons
	train_df = pd.read_csv(OUT_TRAIN, sep='\t').sample(frac=.35, random_state=SEED)
	test_df = pd.read_csv(OUT_TEST, sep='\t').sample(frac=.35, random_state=SEED)

	# distribution visualization of test set
	distrib_visualization(test_df, eval_outdir+'test-dataset-distrib.png')
	# distribution visualization of train set BEFORE normalization
	distrib_visualization(train_df, eval_outdir+'train-dataset-distrib-before-normalization.png')

	# undersampling
	# get all values == to the lowest occurring value in train dataset (ELDR)
	train_df = normalize_data_distrib(train_df, 'Verts-ALE')
	train_df = normalize_data_distrib(train_df, 'GUE-NGL')

	# bias = 50% of the # of training data for the baseline category (ELDR)
	bias = round(len(train_df[train_df.parti == 'ELDR'])*0.5)

	# train_df = normalize_data_distrib(train_df, 'PPE-DE', b=bias)
	# train_df = normalize_data_distrib(train_df, 'PSE', b=bias)
	train_df = normalize_data_distrib(train_df, 'PPE-DE')
	train_df = normalize_data_distrib(train_df, 'PSE')


	# distribution visualization of train set AFTER normalization
	distrib_visualization(train_df, eval_outdir+'train-dataset-distrib-after-normalization.png')

	X_train, y_train = cm.fit_X_y_train(train_df)
	# train model
	cm.train_model(X_train,y_train)

	# load last saved model
	print('Loading model...')
	model = pickle.load(open(cm.model_filename, 'rb'))

	X_test, y_test = cm.fit_X_y_test(test_df)
	y_pred = cm.get_model_preds(model, X_test)
	accuracy = cm.evaluate(model, X_test, y_test)
	labels = model.classes_

	# get model accuracy
	print('Accuracy on test set is {}%.'.format(accuracy))

	print('Saving Confusion Matrix...')
	matrix = confusion_matrix(y_test,y_pred,labels=labels)
	plot = metrics.ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
	plot.plot()
	plt.savefig(eval_outdir+'confusion_matrix.png')

	print(f'Creating classification report...')
	cr = classification_report(y_test,y_pred,target_names=labels)
	open(eval_outdir+'classification-report.txt', 'w').write(f'{cr}')

	# visualize some of the predictions of the model on n random examples found in the test set
	n = 50

	test_features = test_df.doc.head(n).to_list()
	test_labels = test_df.parti.head(n).to_list()

	preds = cm.predict_new_input(model, *test_features)
	preds_df = pd.DataFrame({'text':['']*n, 'prediction':['']*n, 'true':['']*n})

	for i, pred in enumerate(preds):
		preds_df.text[i] = test_features[i]
		preds_df.prediction[i] = pred
		preds_df.true[i] = test_labels[i]

	excel_outfile = eval_outdir+'test_model_predictions.xlsx'
	preds_df.to_excel(excel_outfile)