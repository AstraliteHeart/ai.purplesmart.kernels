
import asyncio
import concurrent.futures
import functools

import runner
from ray import serve
from transformers import GPT2Tokenizer


def run_in_executor(f):
    @functools.wraps(f)
    def inner(*args, **kwargs):
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return loop.run_in_executor(None, functools.partial(f, *args, **kwargs))

    return inner

class Tokenizer(runner.Runner):
    name = 'tokenizer'

    def __init__(self, counter, name, max_concurrent_queries):
        super().__init__(counter, name, max_concurrent_queries)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
       
    @run_in_executor
    def get_results(self, text):
        return self.tokenizer(text)["input_ids"]
        
    async def func(self, request, **kwargs):
        try:
            data = await self.get_results(request)
        except:
            self.log.critical('Failure', exc_info=1)
            data = None
        return data
