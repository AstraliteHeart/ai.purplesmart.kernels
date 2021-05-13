import os

from ray import serve
from typing import Optional

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import pipeline, set_seed
import concurrent.futures

from ray import serve
import runner
import functools
import asyncio

from pydantic import BaseModel

MAX_LENGTH = 700
DEFAULT_TEMPERATURE = 0.85
DEFAULT_TOP_P = 0.90
DEFAULT_TOP_K = 40


class RequestSettings(BaseModel):
    min_length: Optional[int]
    max_length = 20
    top_k = 50
    do_sample = True
    top_p = 0.95
    num_return_sequences = 1
    temperature: Optional[float]
    repetition_penalty: Optional[float]
    num_beams: Optional[int] = 1
    no_repeat_ngram_size: Optional[int]
    remove_prompt: Optional[bool]


def run_in_executor(f):
    @functools.wraps(f)
    def inner(*args, **kwargs):
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return loop.run_in_executor(None, functools.partial(f, *args, **kwargs))

    return inner


@serve.deployment(name="generate", version="1", route_prefix="/generate")
class Generate(runner.Runner):
    name = "generation"

    def __init__(self, counter, name, max_concurrent_queries):
        super().__init__(counter, name, max_concurrent_queries)

        self.log.debug("Loading model")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
        self.model = GPT2LMHeadModel.from_pretrained(
            os.path.join(
                self.data_dir,
                "generate",
                "v1",
                "2020-08-20-astraliteheart-gpt215b-sffuberset",
            ),
            pad_token_id=self.tokenizer.eos_token_id,
        )
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device_id,
        )
        set_seed(42)
        self.log.debug("Model loaded")

    @run_in_executor
    def get_results(self, text, **kwargs):
        encoded_len = len(self.tokenizer(text)["input_ids"])
        self.log.info("encoded_len %s", encoded_len)

        settings = RequestSettings(**kwargs)
        self.log.info("settings %s", settings.dict())

        if settings.max_length:
            settings.max_length += encoded_len
            self.log.info("max_length 1 %s", settings.max_length)
        else:
            settings.max_length += min(MAX_LENGTH, (MAX_LENGTH - 100) - encoded_len)
            self.log.info("max_length 1 %s", settings.max_length)
        if not settings.min_length:
            settings.min_length = encoded_len + 10

        self.log.debug("max_length %s", settings.max_length)
        data = self.pipeline(text, **settings.dict(exclude={"remove_prompt": ...}))
        if settings.remove_prompt:
            for entry in data:
                self.log.info("before %s", entry["generated_text"])
                entry["generated_text"] = entry["generated_text"][len(text) :]
                self.log.info("after %s", entry["generated_text"])

        return data

    async def func(self, request, **kwargs):
        try:
            data = await self.get_results(request, **kwargs)
        except:
            self.log.critical("Failure", exc_info=1)
            data = None
        return data
