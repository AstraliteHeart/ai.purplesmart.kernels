import runner
from ray import serve
from transformers import pipeline

class Summarize(runner.Runner):
    name = "summarize"

    def __init__(self, counter, name, max_concurrent_queries):
        super().__init__(counter, name, max_concurrent_queries)
        self.nlp_model = pipeline("summarization", device=self.device_id)

    async def func(self, request, max_length=100):
        return self.nlp_model(request, max_length=max_length)
