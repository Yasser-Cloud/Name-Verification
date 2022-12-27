import os
import json
from datetime import datetime
from typing import Dict, List, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model

class Trainer():

    def __init__(self) -> None:
        self.__storage_path = os.path.join(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))), 'storage')
        if not os.path.exists(self.__storage_path):
            os.mkdir(self.__storage_path)
        self.__status_path = os.path.join(
            self.__storage_path, 'model_status.json')
        self.__model_path = os.path.join(
            self.__storage_path, 'best_model.h5')
        self.__token_path = os.path.join(
            self.__storage_path, 'tokenizer.json')
        self.__max_sequence_len = 5

        if os.path.exists(self.__status_path):
            with open(self.__status_path) as file:
                self.model_status = json.load(file)
        else:
            self.model_status = {"status": "No Model found",
                                 "timestamp": datetime.now().isoformat(), "classes": []}

        if os.path.exists(self.__model_path):
            self.model = load_model(self.__model_path)
            self._update_status("Model Ready",['Real','Fake'])
        else:
            self.model = None

        if os.path.exists(self.__token_path):
            with open(self.__token_path) as file:
                self.__tokenizer = Tokenizer()
                self.__tokenizer.word_index = json.load(file)
        else:
                self.__tokenizer = Tokenizer()



    def predict(self, texts: List[str]) -> List[Dict]:
        response = []
        if self.model:
            for line in texts:
                  start = datetime.now()
                  token = self.__tokenizer.texts_to_sequences([line])[0]
                  row =pad_sequences([token], maxlen=self.__max_sequence_len, padding='post')

                  probs = self.model.predict(row)
                  prob = probs[0][0]
                  row_pred = {}
                  row_pred['text'] = line
                  row_pred['predictions'] = {'Real': round(float(prob), 3),
                  'Fake': round(float(1-prob), 3)
                  }
                  row_pred['evaluation'] = "real name with low confidence " + str(float(prob)) if round(float(prob), 3) < 0.5 \
                  else "real name with high confidence "+ str(float(prob))
                  row_pred['executiontime'] = str(datetime.now() - start)
                  response.append(row_pred)            
        else:
            raise Exception("No Trained model was found.")
        return response

    def get_status(self) -> Dict:
        return self.model_status

    def _update_status(self, status: str, classes: List[str] = []) -> None:
        self.model_status['status'] = status
        self.model_status['timestamp'] = datetime.now().isoformat()
        self.model_status['classes'] = classes

        with open(self.__status_path, 'w+') as file:
            json.dump(self.model_status, file, indent=2)
