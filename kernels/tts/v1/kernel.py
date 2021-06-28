import base64
import logging
import os
import sys

import runner
from ray import serve

from pathlib import Path

from pydantic import BaseModel

class RequestSettings(BaseModel):
    speaker: str = "(Show) My Little Pony_Twilight"


WG_CURRENT = "LargeWaveGlow V3.5 (GT, 48 Flow, n_group 24 190K)"
TT_CURRENT = "Tacotron2 Torchmoji v0.22.1 (Large Prenet 188K)" 

class TTS(runner.Runner):
    name = "tts"

    def __init__(self, counter, name, max_concurrent_queries):
        super().__init__(counter, name, max_concurrent_queries)

        sys.path.append(os.path.join(self.data_dir, "tacotron2-PPP-1.3.0"))

        from text2speech import T2S

        self.t2s = T2S(
            self.device_id,
            os.path.join(
                self.data_dir, "tts", "v1", "2021-03-14-astraliteheart-tts-mlp"
            ),
        )

        self.max_input_length = self.t2s.conf["html_max_input_len"]
        self.sample_max_attempts = 96
        self.sample_max_duration_s = 20
        self.sample_batch_size = 96
        self.sample_dyna_max_duration_s = 0.125
        self.sample_target_score = 0.75
        self.sample_multispeaker_mode = "random"
        self.sample_cat_silence_s = 0.1
        self.sample_textseg_len_target = 120

    async def func_internal(self, request, tmpdirname, **kwargs):     
        emoji_score = None

        settings = RequestSettings(**kwargs)

        try:
            speaker = [settings.speaker]
            style_mode = "torchmoji_hidden"  # task_context.input_style_mode
            textseg_mode = "segment_by_sentencequote"  # task_context.input_textseg_mode
            batch_mode = "nochange"  # task_context.input_batch_mode
            max_attempts = self.sample_max_attempts
            max_duration_s = self.sample_max_duration_s
            batch_size = self.sample_batch_size
            dyna_max_duration_s = self.sample_dyna_max_duration_s
            use_arpabet = True  # if task_context.input_use_arpabet == "on" else False
            target_score = self.sample_target_score
            multispeaker_mode = self.sample_multispeaker_mode

            # update tacotron if needed
            if self.t2s.tt_current != TT_CURRENT:
                self.t2s.update_tt(TT_CURRENT)

            # update waveglow if needed
            if self.t2s.wg_current != WG_CURRENT:
                self.t2s.update_wg(WG_CURRENT)

            # (Text) CRLF to LF
            text = request.replace("\r\n", "\n")

            # (Text) Max Lenght Limit
            text = text[: int(self.max_input_length)]

            # generate an audio file from the inputs
            (
                out_file,
                gen_time,
                gen_dur,
                total_specs,
                n_passes,
                avg_score,
                arpa_text,
                emoji_score,
            ) = self.t2s.infer(
                text,
                speaker,
                style_mode,
                textseg_mode,
                batch_mode,
                max_attempts,
                max_duration_s,
                batch_size,
                dyna_max_duration_s,
                use_arpabet,
                target_score,
                multispeaker_mode,
                self.sample_cat_silence_s,
                self.sample_textseg_len_target,
            )

            data_out = Path(out_file).read_bytes()
            generated_text = arpa_text
            generated_data = base64.b64encode(data_out).decode("ascii")
            """
            if task_context.base64:
                generated_data = base64.b64encode(data_out).decode("ascii")
            else:
                generated_data = data_out
            """
        except:
            logging.exception("Generation error")
            generated_text = "Generation error"
            generated_data = ""

        return {
            "generated_text": generated_text,
            "generated_data": generated_data,
            "emoji_score": emoji_score,
        }, out_file

    async def func(self, request, **kwargs):
        emoji_score = None

        settings = RequestSettings(**kwargs)

        try:
            speaker = [settings.speaker]
            style_mode = "torchmoji_hidden"  # task_context.input_style_mode
            textseg_mode = "segment_by_sentencequote"  # task_context.input_textseg_mode
            batch_mode = "nochange"  # task_context.input_batch_mode
            max_attempts = self.sample_max_attempts
            max_duration_s = self.sample_max_duration_s
            batch_size = self.sample_batch_size
            dyna_max_duration_s = self.sample_dyna_max_duration_s
            use_arpabet = True  # if task_context.input_use_arpabet == "on" else False
            target_score = self.sample_target_score
            multispeaker_mode = self.sample_multispeaker_mode

            # update tacotron if needed
            if self.t2s.tt_current != TT_CURRENT:
                self.t2s.update_tt(TT_CURRENT)

            # update waveglow if needed
            if self.t2s.wg_current != WG_CURRENT:
                self.t2s.update_wg(WG_CURRENT)

            # (Text) CRLF to LF
            text = request.replace("\r\n", "\n")

            # (Text) Max Lenght Limit
            text = text[: int(self.max_input_length)]

            # generate an audio file from the inputs
            (
                out_file,
                gen_time,
                gen_dur,
                total_specs,
                n_passes,
                avg_score,
                arpa_text,
                emoji_score,
            ) = self.t2s.infer(
                text,
                speaker,
                style_mode,
                textseg_mode,
                batch_mode,
                max_attempts,
                max_duration_s,
                batch_size,
                dyna_max_duration_s,
                use_arpabet,
                target_score,
                multispeaker_mode,
                self.sample_cat_silence_s,
                self.sample_textseg_len_target,
            )

            data_out = Path(out_file).read_bytes()
            #data_out = out_file
            generated_text = arpa_text
            generated_data = base64.b64encode(data_out).decode("ascii")
            """
            if task_context.base64:
                generated_data = base64.b64encode(data_out).decode("ascii")
            else:
                generated_data = data_out
            """
        except:
            logging.exception("Generation error")
            generated_text = "Generation error"
            generated_data = ""

        return {
            "generated_text": generated_text,
            "generated_data": generated_data,
            "emoji_score": emoji_score,
        }
