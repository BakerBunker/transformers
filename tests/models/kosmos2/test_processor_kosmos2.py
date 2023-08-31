# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import shutil
import tempfile
import unittest

import numpy as np
import pytest

from transformers.testing_utils import get_tests_dir, require_vision, require_sentencepiece, require_tokenizers
from transformers.utils import is_vision_available


if is_vision_available():
    from PIL import Image

    from transformers import AutoProcessor, BertTokenizer, Kosmos2ImageProcessor, Kosmos2Processor, PreTrainedTokenizerFast, Kosmos2Tokenizer, Kosmos2TokenizerFast


SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")


@require_sentencepiece
@require_tokenizers
@require_vision
class Kosmos2ProcessorTest(unittest.TestCase):
    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()

        image_processor = Kosmos2ImageProcessor()

        # We have a SentencePiece fixture for testing
        slow_tokenizer = Kosmos2Tokenizer(SAMPLE_VOCAB)
        fast_tokenizer = Kosmos2TokenizerFast(__slow_tokenizer=slow_tokenizer)

        processor = Kosmos2Processor(image_processor, fast_tokenizer)
        processor.save_pretrained(self.tmpdirname)

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def prepare_image_inputs(self):
        """This function prepares a list of PIL images, or a list of numpy arrays if one specifies numpify=True,
        or a list of PyTorch tensors if one specifies torchify=True.
        """

        image_inputs = [np.random.randint(255, size=(3, 30, 400), dtype=np.uint8)]

        image_inputs = [Image.fromarray(np.moveaxis(x, 0, -1)) for x in image_inputs]

        return image_inputs

    def test_save_load_pretrained_additional_features(self):
        processor = Kosmos2Processor(tokenizer=self.get_tokenizer(), image_processor=self.get_image_processor())
        processor.save_pretrained(self.tmpdirname)

        tokenizer_add_kwargs = self.get_tokenizer(bos_token="(BOS)", eos_token="(EOS)")
        image_processor_add_kwargs = self.get_image_processor(do_normalize=False, padding_value=1.0)

        processor = Kosmos2Processor.from_pretrained(
            self.tmpdirname, bos_token="(BOS)", eos_token="(EOS)", do_normalize=False, padding_value=1.0
        )

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())
        self.assertIsInstance(processor.tokenizer, PreTrainedTokenizerFast)

        self.assertEqual(processor.image_processor.to_json_string(), image_processor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.image_processor, Kosmos2ImageProcessor)

    def test_image_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = Kosmos2Processor(tokenizer=tokenizer, image_processor=image_processor)

        image_input = self.prepare_image_inputs()

        input_feat_extract = image_processor(image_input, return_tensors="np")
        input_processor = processor(images=image_input, return_tensors="np")

        for key in input_feat_extract.keys():
            self.assertAlmostEqual(input_feat_extract[key].sum(), input_processor[key].sum(), delta=1e-2)

    def test_tokenizer(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = Kosmos2Processor(tokenizer=tokenizer, image_processor=image_processor)

        input_str = "This is a test"

        encoded_processor = processor(text=input_str)

        encoded_tok = tokenizer(input_str, return_token_type_ids=False)

        for key in encoded_tok.keys():
            self.assertListEqual(encoded_tok[key], encoded_processor[key])

    def test_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = Kosmos2Processor(tokenizer=tokenizer, image_processor=image_processor)

        input_str = "This is a test"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input)

        self.assertListEqual(list(inputs.keys()), ["pixel_values", "input_ids", "attention_mask", "image_features_mask"])

        # test if it raises when no input is passed
        with pytest.raises(ValueError):
            processor()

    def test_tokenizer_decode(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = Kosmos2Processor(tokenizer=tokenizer, image_processor=image_processor)

        predicted_ids = [[1, 4, 5, 8, 1, 0, 8], [3, 4, 3, 1, 1, 8, 9]]

        decoded_processor = processor.batch_decode(predicted_ids)
        decoded_tok = tokenizer.batch_decode(predicted_ids)

        self.assertListEqual(decoded_tok, decoded_processor)

    def test_model_input_names(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = Kosmos2Processor(tokenizer=tokenizer, image_processor=image_processor)

        input_str = "This is a test"
        image_input = self.prepare_image_inputs()

        # both image and text
        inputs = processor(text=input_str, images=image_input)
        self.assertListEqual(list(inputs.keys()), ["pixel_values", "input_ids", "attention_mask", "image_features_mask"])

        # only text
        inputs = processor(text=input_str)
        self.assertListEqual(list(inputs.keys()), ["input_ids", "attention_mask"])

        # only image
        inputs = processor(images=image_input)
        self.assertListEqual(list(inputs.keys()), ["pixel_values"])

    def test_full_processor(self):

        processor = Kosmos2Processor.from_pretrained("ydshieh/temp-testing-kosmos-2")

        # test with different input formats.
        # fmt: off
        texts = [
            # no phrase
            "<grounding> Two puppies sit in a field of grass.",
            # 1 phrase
            "<grounding> <phrase> Two puppies </phrase> sit in a field of grass.",
            # 2 phrases
            "<grounding> <phrase> Two puppies </phrase> sit in a field of <phrase> grass </phrase>.",
        ]
        # fmt: on

        # TODO: add to the official repo.
        image = "https://huggingface.co/ydshieh/kosmos-2-patch14-224/resolve/main/two_dogs.jpg"

        # fmt: off
        expected_texts = [
            # no phrase
            "<grounding> Two puppies sit in a field of grass.",
            # 1 phrase: without bbox
            "<grounding><phrase> Two puppies</phrase> sit in a field of grass.",
            # 1 phrase: with a single bbox
            "<grounding><phrase> Two puppies</phrase><object><patch_index_0079><patch_index_1016></object> sit in a field of grass.",  # noqa
            # 1 phrase: with 2 bboxes
            "<grounding><phrase> Two puppies</phrase><object><patch_index_0079><patch_index_1016></delimiter_of_multi_objects/><patch_index_0135><patch_index_1008></object> sit in a field of grass.",  # noqa
            # 2 phrases: one with 2 bboxes and another one without bbox
            "<grounding><phrase> Two puppies</phrase><object><patch_index_0079><patch_index_1016></delimiter_of_multi_objects/><patch_index_0135><patch_index_1008></object> sit in a field of<phrase> grass</phrase> .",  # noqa
            # 2 phrases: one with 2 bboxes and another one with a single bbox
            "<grounding><phrase> Two puppies</phrase><object><patch_index_0079><patch_index_1016></delimiter_of_multi_objects/><patch_index_0135><patch_index_1008></object> sit in a field of<phrase> grass</phrase><object><patch_index_0480><patch_index_1023></object> .",  # noqa
        ]
        # fmt: on

        # no phrase
        a = processor.preprocess_text(images=None, texts=texts[0], bboxes=None)
        assert a == expected_texts[0]

        # no phrase
        a = processor.preprocess_text(images=None, texts=texts[0], bboxes=[])
        assert a == expected_texts[0]

        # 1 phrase: no bbox
        a = processor.preprocess_text(images=None, texts=texts[1], bboxes=[None])
        assert a == expected_texts[1]

        # 1 phrase: no bbox
        a = processor.preprocess_text(images=None, texts=texts[1], bboxes=[[]])
        assert a == expected_texts[1]

        # a = processor.preprocess_text(images=None, texts=texts[1], bboxes=[[None]])
        # assert a == expected_texts[1]

        # 1 phrase: 1 bbox
        a = processor.preprocess_text(images=None, texts=texts[1], bboxes=[(79, 1016)])
        assert a == expected_texts[2]

        # 1 phrase: 1 bbox
        a = processor.preprocess_text(images=None, texts=texts[1], bboxes=[[(79, 1016)]])
        assert a == expected_texts[2]

        # 1 phrase: 2 bboxes
        a = processor.preprocess_text(images=None, texts=texts[1], bboxes=[[(79, 1016), (135, 1008)]])
        assert a == expected_texts[3]

        # 2 phrase: 2 bboxes + no bbox
        a = processor.preprocess_text(images=None, texts=texts[2], bboxes=[[(79, 1016), (135, 1008)], None])
        assert a == expected_texts[4]

        # 2 phrase: 2 bboxes + no bbox
        a = processor.preprocess_text(images=None, texts=texts[2], bboxes=[[(79, 1016), (135, 1008)], []])
        assert a == expected_texts[4]

        # a = processor.preprocess_text(images=None, texts=texts[2], bboxes=[[(79, 1016), (135, 1008)], [None]])
        # assert a == expected_texts[4]

        # 2 phrase: 2 bboxes + 1 bbox
        a = processor.preprocess_text(images=None, texts=texts[2], bboxes=[[(79, 1016), (135, 1008)], (480, 1023)])
        assert a == expected_texts[5]

        # 2 phrase: 2 bboxes + 1 bbox
        a = processor.preprocess_text(images=None, texts=texts[2], bboxes=[[(79, 1016), (135, 1008)], [(480, 1023)]])
        assert a == expected_texts[5]

        # batch
        a = processor.preprocess_text(
            images=None,
            texts=[texts[0], texts[1], texts[1], texts[2]],
            bboxes=[
                None,  # no phrase
                [[]],  # 1 phrase: no bbox
                [(79, 1016)],  # 1 phrase: 1 bbox
                [[(79, 1016), (135, 1008)], (480, 1023)],  # 2 phrase: 2 bboxes + 1 bbox
            ]
        )
        assert a == [expected_texts[0], expected_texts[1], expected_texts[2], expected_texts[5]]
