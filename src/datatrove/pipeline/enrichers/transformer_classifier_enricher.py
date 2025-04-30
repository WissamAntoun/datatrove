from typing import List

from datatrove.data import Document
from datatrove.pipeline.enrichers.base_enricher import BaseEnricher
from datatrove.utils.text import SPLIT_TEXT_DOCUMENTS, split_into_parts


def default_pre_process(docs: List[Document]):
    return docs


def url_injection_pre_process(docs: List[Document]):
    # return {"texts": [doc.metadata["url"] + "\n\n" + doc.text for doc in docs]}
    preped_docs = []
    for doc in docs:
        doc.text = doc.metadata.get("url", "") + "\n\n" + doc.text
        preped_docs.append(doc)
    return preped_docs


PREPROCESSORS = {
    "default": default_pre_process,
    "url_injection": url_injection_pre_process,
}


class TransformerClassifierEnricher(BaseEnricher):
    """Adds the output of a Transformer classifier to the metadata of the document.

    Args:
        model_name: name of the model to use
        field_name: field name to use for the classification metadata
        split_mode: predict and filter on DOCUMENT, PARAGRAPH or SENTENCE level
        store_units: store the units in the metadata
        kwargs: additional arguments to pass to the TextClassificationPipeline
    """

    name = "🤖 Transformer Enricher"
    _requires_dependencies = ["transformers"]

    def __init__(
        self,
        model_name_or_path: str,
        field_name: str,
        split_mode: str = SPLIT_TEXT_DOCUMENTS,
        store_units: str = False,
        batch_size: int = 1,
        model_batch_size: int = None,
        sort_batch_by_length: bool = False,
        preprocess_fn: str = "default",
        num_gpus: int = 1,
        pipeline_kwargs: dict = None,
        model_kwargs: dict = None,
        tokenizer_kwargs: dict = None,
        call_kwargs: dict = None,
    ):
        super().__init__(batch_size)
        self.model_name_or_path = model_name_or_path
        self.field_name = field_name
        self.split_mode = split_mode
        self.store_units = store_units
        self.sort_batch_by_length = sort_batch_by_length
        self.model_batch_size = model_batch_size if model_batch_size else batch_size
        self.pre_process = PREPROCESSORS[preprocess_fn]
        self._model = None
        self.num_gpus = num_gpus
        self.pipeline_kwargs = pipeline_kwargs if pipeline_kwargs else {}
        self.model_kwargs = model_kwargs if model_kwargs else {}
        self.tokenizer_kwargs = tokenizer_kwargs if tokenizer_kwargs else {}
        self.call_kwargs = call_kwargs if call_kwargs else {}

        if self.num_gpus > 1:
            print(f"Warning: Using multiple GPUs ({self.num_gpus}) for inference.")

    @property
    def model(self):
        if self._model is None:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
                TextClassificationPipeline,
            )

            if self.num_gpus == 1:
                self._model = TextClassificationPipeline(
                    model=AutoModelForSequenceClassification.from_pretrained(
                        self.model_name_or_path,
                        **self.model_kwargs,
                    ),
                    tokenizer=AutoTokenizer.from_pretrained(
                        self.model_name_or_path,
                        **self.tokenizer_kwargs,
                    ),
                    **self.pipeline_kwargs,
                )
            else:
                self._model = [
                    TextClassificationPipeline(
                        model=AutoModelForSequenceClassification.from_pretrained(
                            self.model_name_or_path,
                            **self.model_kwargs,
                        ),
                        tokenizer=AutoTokenizer.from_pretrained(
                            self.model_name_or_path,
                            **self.tokenizer_kwargs,
                        ),
                        **self.pipeline_kwargs,
                        device=i,
                    )
                    for i in range(self.num_gpus)
                ]

        return self._model

    def _do_inference(self, batch, model):
        batch = self.pre_process(batch)

        text_batch = []
        batch_id_to_text_batch_id_map = {}
        for idx, doc in enumerate(batch):
            units = split_into_parts(doc.text, mode=self.split_mode)
            batch_id_to_text_batch_id_map[idx] = [idx + i for i in range(len(units))]
            text_batch.extend(units)

        if self.sort_batch_by_length:
            # sort batch by length and maintain a mapping dict to the original order
            ibatch = list(enumerate(text_batch))  # [ (0, doc0), (1, doc1), (2, doc2), ...]
            sbatch = sorted(ibatch, key=lambda x: len(x[1]), reverse=True)  # [(2, doc2), (0, doc0), (1, doc1), ...]
            sbatch_data = [x[1] for x in sbatch]  # [doc2, doc0, doc1, ...]
            sbatch_mapping = {x[0]: i for i, x in enumerate(sbatch)}  # {0: 2, 1: 0, 2: 1}
        else:
            sbatch_data = text_batch

        # Do the actual classification
        scores = model(sbatch_data, **self.call_kwargs)

        if self.sort_batch_by_length:
            # sort back to original order
            scores_orig_batch_order = [scores[sbatch_mapping[i]] for i in range(len(sbatch_data))]
        else:
            scores_orig_batch_order = scores

        for idx, doc in enumerate(batch):
            label_scores = []
            for text_id in batch_id_to_text_batch_id_map[idx]:
                _label_scores = {}
                _label_scores["score"] = scores_orig_batch_order[text_id]
                if self.store_units:
                    _label_scores["unit"] = text_batch[text_id]
                label_scores.append(_label_scores)
            doc.metadata[self.field_name] = label_scores

        return batch

    def enrich_batch(self, batch: List[Document]) -> List[Document]:
        if self.num_gpus == 1:
            batch = self._do_inference(batch)
        else:
            indexed_batch = list(enumerate(batch))

            per_gpu_batch = [indexed_batch[i :: self.num_gpus] for i in range(self.num_gpus)]

            from concurrent.futures import ThreadPoolExecutor

            def run_pipeline_with_index(pipe, indexed_inputs):
                results = self._do_inference([text for idx, text in indexed_inputs], pipe)
                return [(indexed_inputs[i][0], result) for i, result in enumerate(results)]

            with ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
                results = list(executor.map(run_pipeline_with_index, self.model, per_gpu_batch))

            # Flatten the results
            results = [item for sublist in results for item in sublist]
            # Sort the results back to the original order
            results.sort(key=lambda x: x[0])
            # Unpack the results
            batch = [result[1] for result in results]

        return batch

    def enrich(self, doc: Document) -> Document:
        units = split_into_parts(doc.text, mode=self.split_mode)

        self.stat_update("doc-total")
        self.stat_update("units", value=len(units), unit="doc")

        label_scores = []
        for unit in units:
            scores = self.model(unit)
            _label_scores = {}
            _label_scores["score"] = scores
            if self.store_units:
                _label_scores["unit"] = unit
            label_scores.append(_label_scores)

        doc.metadata[self.field_name] = label_scores
        return doc
