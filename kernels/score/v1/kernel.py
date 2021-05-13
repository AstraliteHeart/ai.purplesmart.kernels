import runner
import torch
from ray import serve
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          set_seed)

MODEL_CARD = "microsoft/DialogRPT-human-vs-rand"

@serve.deployment(name="score", version="1")
class Score(runner.Runner):
    name = "score"

    def __init__(self, counter, name, max_concurrent_queries):
        super().__init__(counter, name, max_concurrent_queries)
        self.device = torch.device(f"cuda:{self.device_id}")
        self.log.debug("Loading model")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_CARD)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_CARD).to(
            self.device
        )
        set_seed(42)
        self.log.debug("Model loaded")

    async def func(self, request, continuation=""):
        try:
            model_input = self.tokenizer.encode(
                request + "<|endoftext|>" + continuation, return_tensors="pt"
            ).to(self.device)
            result = self.model(model_input, return_dict=True)
            return float(torch.sigmoid(result.logits).squeeze())
        except:
            self.log.critical("Failure", exc_info=1)
            data = None
        return data
