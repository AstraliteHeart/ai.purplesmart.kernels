import json
import os
import sys

import emoji
import numpy as np
import runner
import torch
from ray import serve

EMOJIS = ":joy: :unamused: :weary: :sob: :heart_eyes: :pensive: :ok_hand: :blush: :heart: :smirk: :grin: :notes: :flushed: :100: :sleeping: :relieved: :relaxed: :raised_hands: :two_hearts: :expressionless: :sweat_smile: :pray: :confused: :kissing_heart: :heartbeat: :neutral_face: :information_desk_person: :disappointed: :see_no_evil: :tired_face: :v: :sunglasses: :rage: :thumbsup: :cry: :sleepy: :yum: :triumph: :hand: :mask: :clap: :eyes: :gun: :persevere: :smiling_imp: :sweat: :broken_heart: :yellow_heart: :musical_note: :speak_no_evil: :wink: :skull: :confounded: :smile: :stuck_out_tongue_winking_eye: :angry: :no_good: :muscle: :facepunch: :purple_heart: :sparkling_heart: :blue_heart: :grimacing: :sparkles:".split(
    " "
)
MAX_LEN = 130


def top_elements(array, k):
    ind = np.argpartition(array, -k)[-k:]
    return ind[np.argsort(array[ind])][::-1]


@serve.deployment(name="emoji", version="1")
class Emoji(runner.Runner):
    name = "emoji"

    def __init__(self, counter, name, max_concurrent_queries):
        super().__init__(counter, name, max_concurrent_queries)

        sys.path.append(os.path.join(self.data_dir, "tacotron2-PPP-1.3.0"))
        from torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH
        from torchmoji.model_def import torchmoji_emojis, torchmoji_feature_encoding
        from torchmoji.sentence_tokenizer import SentenceTokenizer

        self.log.debug("Loading model")

        with open(VOCAB_PATH, "r") as f:
            vocabulary = json.load(f)

        with torch.no_grad():
            self.tm_sentence_tokenizer = SentenceTokenizer(
                vocabulary, MAX_LEN, ignore_sentences_with_only_custom=True
            )
            self.tm_torchmoji = torchmoji_feature_encoding(PRETRAINED_PATH)
            self.tm_model = torchmoji_emojis(PRETRAINED_PATH)

        self.log.debug("Model loaded")

    async def func(self, request, **kwargs):
        text_batch = [self.normalize_input(request)]
        text_batch = [
            text.replace('"', "") for text in text_batch
        ]  # remove quotes from text
        tokenized, _, _ = self.tm_sentence_tokenizer.tokenize_sentences(text_batch)
        prob = self.tm_model(tokenized)[0]
        emoji_ids = top_elements(prob, 3)
        emojis = map(lambda x: EMOJIS[x], emoji_ids)
        emoji_score = [emoji.emojize(e, use_aliases=True) for e in emojis]
        return emoji_score
