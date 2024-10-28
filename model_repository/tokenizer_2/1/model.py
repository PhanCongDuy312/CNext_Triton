

import os
from typing import Dict, List
from transformers import  CLIPTokenizer
from pathlib import Path
from typing import Callable, List, Optional, Union, Dict
import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer, PreTrainedTokenizer, TensorType


class TritonPythonModel:
    def initialize(self, args: Dict[str, str]):
        current_name: str = str(Path(args["model_repository"]).parent.absolute())

        self.tokenizer = CLIPTokenizer.from_pretrained(
            current_name + "/tokenizer_2/1/config/"
        )
  
    def execute(self, requests) -> "List[List[pb_utils.Tensor]]":
        """
        Parse and tokenize each request
        :param requests: 1 or more requests received by Triton server.
        :return: text as input tensors
        """
        responses = []
        # TODO: update to process batch requests
        for request in requests:
            # binary data typed back to string
            query = [
                t.decode("UTF-8")
                for t in pb_utils.get_input_tensor_by_name(request, "TEXT")
                .as_numpy()
                .tolist()
            ]

            # tokenization
            text_input_ids = self.tokenizer(
                query,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="np",
            ).input_ids.astype(np.int32)

            # raw tokenization without truncation
            untruncated_ids = self.tokenizer(
                query, 
                padding="max_length", 
                return_tensors="np"
            ).input_ids.astype(np.int32)

            # only for logging
            if not np.array_equal(text_input_ids, untruncated_ids):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                self.logger.log_warn(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            # communicate the tokenization results to Triton server
            tensor_output = pb_utils.Tensor('input_ids', text_input_ids)
            inference_response = pb_utils.InferenceResponse(output_tensors=[tensor_output])
            responses.append(inference_response)

        return responses