import os
import warnings
import json
from typing import Dict
from word2number.w2n import word_to_num
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from transformers import BertTokenizerFast, AutoModelForTokenClassification, pipeline 

warnings.filterwarnings("ignore")

class NLPManager:
	def __init__(self):
		tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
		model_fine_tuned = AutoModelForTokenClassification.from_pretrained("ner_model")

		self.nlp = pipeline("ner", model=model_fine_tuned, tokenizer=tokenizer)

	def qa(self, context: str) -> Dict[str, str]:
        # perform NLP question-answering
		ans = self.recog(context)
		return {"heading": f"{ans['HDG']}", "tool": f"{ans['TOL']}", "target": f"{ans['TAR']}"}

	def recog(self, context:str):
		ner_results = self.nlp(context)

		word_lst = []
		word_dict = {'TOL': None, 'HDG': None, 'TAR': None}

		prev = ner_results[0]['entity'][2:]
		word = ner_results[0]['word']

		for i in range(1, len(ner_results)):
			curr = ner_results[i]['entity'][2:]
			if prev == curr:
				word = word + ' ' +  ner_results[i]['word']
			else:
				word_lst.append(word)
				word_dict[prev] = word_lst[0]
				word_lst = []
				prev = curr
				word = ner_results[i]['word']
				
		word_lst.append(word)
		word_dict[prev] = word_lst[0]
		try:
			word_dict['HDG'] = ''.join(list(map(lambda x: str(word_to_num(x)), word_dict['HDG'].split(' '))))
		except:
			word_dict['HDG'] = '000'
		return word_dict