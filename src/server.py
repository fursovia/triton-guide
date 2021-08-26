from functools import lru_cache

import fastapi
import numpy as np
import tritonclient.grpc as grpc
from fastapi import Depends
from allennlp.data.batch import Batch
from allennlp.models.archival import load_archive
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer

from src.export_model import ARCHIVE_PATH

ARHIVE = load_archive(ARCHIVE_PATH)
READER = ARHIVE.validation_dataset_reader
VOCAB = ARHIVE.model.vocab
TOKENIZER = SpacyTokenizer()


def prepare_inputs(text: str):
    tokenzied = TOKENIZER.tokenize(text)
    instance = READER.text_to_instance(tokenzied)
    READER.apply_token_indexers(instance)
    instances = [instance]

    dataset = Batch(instances)
    dataset.index_instances(VOCAB)
    model_inputs = dataset.as_tensor_dict()['tokens']['tokens']['tokens']
    model_inputs = model_inputs.cpu().numpy().astype(np.int64)

    input_ = grpc.InferInput(name=f'input__0', shape=[1, 64], datatype='INT64')
    input_.set_data_from_numpy(input_tensor=model_inputs)
    output = grpc.InferRequestedOutput(name='output__0')
    return {'inputs': [input_], 'outputs': [output]}


@lru_cache()
def get_client() -> grpc.InferenceServerClient:
    client = grpc.InferenceServerClient(url='localhost:8001', verbose=True)
    return client


def predict_torch(text: str, client: grpc.InferenceServerClient = Depends(get_client)):
    inputs = prepare_inputs(text)
    response = client.infer(model_name='classifier_torch', model_version='1', **inputs)
    probs = response.as_numpy('output__0').tolist()
    return probs


def predict_onnx(text: str, client: grpc.InferenceServerClient = Depends(get_client)):
    inputs = prepare_inputs(text)
    response = client.infer(model_name='classifier_onnx', model_version='1', **inputs)
    probs = response.as_numpy('output__0').tolist()
    return probs


def get_app() -> fastapi.FastAPI:
    """Get fastAPI application"""
    app = fastapi.FastAPI()
    app.add_api_route('/predict_onnx', endpoint=predict_onnx, methods=['POST'])
    app.add_api_route('/predict_torch', endpoint=predict_torch, methods=['POST'])
    return app


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(get_app(), host='0.0.0.0', port=8005)
