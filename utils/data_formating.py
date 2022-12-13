import xml.etree.ElementTree as ET
from tqdm import tqdm
import unicodedata
import csv
from os.path import exists
from utils import IN_TEST, OUT_TEST, IN_TRAIN, OUT_TRAIN, TXT_FILE

def fetch_data(f_type: str, xml_infile: str, outfile: str):

	if f_type not in ['train', 'test']:
		raise ValueError(f"Wrong f_type value : {f_type}. Should be either 'test' or 'train'")

	if not exists(outfile):
		if f_type == 'train':
				features, labels = xml_to_list(f_type, xml_infile)
		elif f_type == 'test':
			features, labels = xml_to_list(f_type, xml_infile)

		xml_to_tsv(outfile, features, labels, f_type)

def xml_to_tsv(outfile: str, features, labels, f_type):

	with open(outfile, 'w', newline='') as f:
		writer = csv.writer(f, delimiter='\t')
		writer.writerow(['doc', 'parti'])
		for doc, parti in zip(features, labels):
			writer.writerow([doc, parti])

def xml_to_list(f_type: str, xml_infile: str):

	if f_type not in ['train', 'test']:
		raise ValueError(f"Wrong f_type value : {f_type}. Should be either 'test' or 'train'")

	txt = open(TXT_FILE, 'r')

	features = []
	labels = []

	root = ET.parse(xml_infile).getroot()

	for i in tqdm(range(100), desc=f'{f_type} set'):
		for doc in root.iter('doc'):
			full_text = []
			for text in doc.iter('p'):
				if text.text is not None:
					text.text = unicodedata.normalize("NFKD", text.text)
					full_text.append(text.text)
				if f_type == 'train':
					for party in doc.iter('EVAL_PARTI'):
						parties = party.find('PARTI').attrib['valeur']

				elif f_type == 'test':
					for line in txt.readlines():
						parties = line.split('\t')[1].strip()
						labels.append(parties)

			features.append(' '.join(full_text))
			
			if f_type == 'train':
				labels.append(parties)

	return features, labels

if __name__ == '__main__':

	fetch_data('test', IN_TEST, OUT_TEST)
	fetch_data('train', IN_TRAIN, OUT_TRAIN)