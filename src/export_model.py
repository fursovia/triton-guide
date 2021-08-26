import torch
from allennlp.nn import util
from allennlp.models import Model
from allennlp.data.batch import Batch
from allennlp.models.archival import load_archive
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from allennlp.common.util import import_module_and_submodules
import_module_and_submodules('allennlp_models')


ARCHIVE_PATH = "/Users/i.fursov/Documents/triton_guide/basic_stanford_sentiment_treebank-2020.06.09.tar.gz"
SAVE_TO = "/Users/i.fursov/Documents/triton_guide/model_registry/classifier_ts/1/model.pt"
ONNX_SAVE_TO = "/Users/i.fursov/Documents/triton_guide/model_registry/classifier_onnx/1/model.onnx"


class TracableModel(torch.nn.Module):
    def __init__(self, model: Model):
        super().__init__()
        self._model = model

    def forward(self, tokens: torch.Tensor):
        inputs = {'tokens': {'tokens': tokens}}
        return self._model.forward(inputs)['probs']


def main(archive_path: str = ARCHIVE_PATH, device: int = -1):
    archive = load_archive(archive_path, cuda_device=device)
    tracable_model = TracableModel(archive.model)
    tokenizer = SpacyTokenizer()

    reader = archive.validation_dataset_reader
    instances = []
    for text in ["a very well-made, funny and entertaining picture.", "shitty picture"]:
        tokenzied = tokenizer.tokenize(text)
        instance = reader.text_to_instance(tokenzied)
        reader.apply_token_indexers(instance)
        instances.append(instance)

    dataset = Batch(instances)
    dataset.index_instances(archive.model.vocab)
    model_inputs = util.move_to_device(dataset.as_tensor_dict(), device)['tokens']['tokens']

    module = torch.jit.trace(tracable_model, example_inputs=model_inputs['tokens'], strict=True, )
    module.save(SAVE_TO)

    torch.onnx.export(
        tracable_model, model_inputs['tokens'], ONNX_SAVE_TO, input_names=['input_1'], output_names=['output_1']
    )


if __name__ == '__main__':
    main()
