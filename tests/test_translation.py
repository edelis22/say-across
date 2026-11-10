from contextlib import nullcontext
import sys
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, Mock, patch

from pragmatics.config import NLLB_REVISION
from pragmatics.translation import NLLB


class NLLBBackendTests(TestCase):
    def test_revision_direction_and_no_silent_truncation(self):
        inputs = MagicMock()
        inputs.input_ids.shape = (1, 9)
        inputs.to.return_value = {"input_ids": inputs.input_ids}
        tokenizer = Mock(return_value=inputs)
        tokenizer.unk_token_id = 0
        tokenizer.eos_token_id = 2
        tokenizer.convert_tokens_to_ids.side_effect = {"eng_Latn": 4, "jpn_Jpan": 5}.__getitem__
        tokenizer.decode.return_value = "おはようございます。"
        model = Mock()
        model.to.return_value.eval.return_value = model
        model.generate.return_value = [[Mock(item=Mock(return_value=2))]]
        tokenizer_loader = Mock(return_value=tokenizer)
        model_loader = Mock(return_value=model)
        modules = {
            "torch": SimpleNamespace(inference_mode=nullcontext),
            "transformers": SimpleNamespace(
                AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_loader),
                AutoModelForSeq2SeqLM=SimpleNamespace(from_pretrained=model_loader),
            ),
        }
        with patch.dict(sys.modules, modules):
            nllb = NLLB()
            self.assertEqual(nllb.translate("Good morning.", "eng_Latn", "jpn_Jpan"),
                             "おはようございます。")
            self.assertEqual(tokenizer.src_lang, "eng_Latn")
            self.assertEqual(model.generate.call_args.kwargs["forced_bos_token_id"], 5)
            self.assertEqual(model_loader.call_args.kwargs["revision"], NLLB_REVISION)
            self.assertFalse(tokenizer.call_args.kwargs["truncation"])
            inputs.input_ids.shape = (1, 513)
            with self.assertRaisesRegex(ValueError, "512"):
                nllb.translate("Long sentence.", "eng_Latn", "jpn_Jpan")
            model.generate.assert_called_once()
            model_loader.assert_called_once()


class SpokenTurnTests(TestCase):
    def test_long_turn_translates_all_sections(self):
        translator = NLLB()
        transcript = " ".join(f"Sentence number {index} is complete." for index in range(65))
        with patch.object(translator, "translate", side_effect=lambda text, *_: text) as backend:
            self.assertEqual(translator.translate_turn(transcript, "eng_Latn", "spa_Latn"), transcript)
            self.assertGreater(backend.call_count, 1)
            self.assertTrue(all(len(call.args[0]) <= 501 for call in backend.call_args_list))
