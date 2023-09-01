# coding=utf-8
# Copyright 2023 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch KOSMOS-2 model. """


import inspect
import os
import tempfile
import unittest

import numpy as np
import requests

from transformers.models.kosmos2.configuration_kosmos2 import Kosmos2TextConfig, Kosmos2VisionConfig
from transformers import Kosmos2Config
from transformers.testing_utils import require_torch, require_torch_gpu, require_vision, slow, torch_device
from transformers.utils import is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    _config_zero_init,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from torch import nn

    from transformers import (
        Kosmos2ForConditionalGeneration,
        Kosmos2ForImageTextRetrieval,
        Kosmos2ForQuestionAnswering,
        Kosmos2Model,
        Kosmos2TextModel,
        Kosmos2VisionModel,
    )
    from transformers.models.kosmos2.modeling_kosmos2 import Kosmos2_PRETRAINED_MODEL_ARCHIVE_LIST


if is_vision_available():
    from PIL import Image

    from transformers import Kosmos2Processor


class Kosmos2VisionModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        image_size=32,
        patch_size=4,
        num_channels=3,
        is_training=True,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        dropout=0.1,
        attention_dropout=0.1,
        initializer_range=1e-10,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.scope = scope

        # in ViT, the seq length equals the number of patches + 1 (we add 1 for the [CLS] token)
        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches + 1

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values

    def get_config(self):
        return Kosmos2VisionConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            initializer_range=self.initializer_range,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


# @require_torch
# class Kosmos2VisionModelTest(ModelTesterMixin, unittest.TestCase):
#     pass


class Kosmos2TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        dropout=0.1,
        attention_dropout=0.1,
        max_position_embeddings=512,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        if input_mask is not None:
            batch_size, seq_length = input_mask.shape
            rnd_start_indices = np.random.randint(1, seq_length - 1, size=(batch_size,))
            for batch_idx, start_index in enumerate(rnd_start_indices):
                input_mask[batch_idx, :start_index] = 1
                input_mask[batch_idx, start_index:] = 0

        config = self.get_config()

        return config, input_ids, input_mask

    def get_config(self):
        return Kosmos2TextConfig(
            vocab_size=self.vocab_size,
            embed_dim=self.hidden_size,
            layers=self.num_hidden_layers,
            attention_heads=self.num_attention_heads,
            ffn_dim=self.intermediate_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            max_position_embeddings=self.max_position_embeddings,
        )

    def create_and_check_model(self, config, input_ids, input_mask):
        model = Kosmos2TextModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_ids, attention_mask=input_mask)
            result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, input_mask = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


# @require_torch
# class Kosmos2TextModelTest(ModelTesterMixin, unittest.TestCase):
#     pass


class Kosmos2ModelTester:
    def __init__(self, parent, text_kwargs=None, vision_kwargs=None, is_training=True):
        if text_kwargs is None:
            text_kwargs = {}
        if vision_kwargs is None:
            vision_kwargs = {}

        self.parent = parent
        self.text_model_tester = Kosmos2TextModelTester(parent, **text_kwargs)
        self.vision_model_tester = Kosmos2VisionModelTester(parent, **vision_kwargs)
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        text_config, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()
        vision_config, pixel_values = self.vision_model_tester.prepare_config_and_inputs()

        config = self.get_config()

        return config, input_ids, attention_mask, pixel_values

    def get_config(self):
        return Kosmos2Config.from_text_vision_configs(
            self.text_model_tester.get_config(), self.vision_model_tester.get_config(),
        )

    def create_and_check_model(self, config, input_ids, attention_mask, pixel_values):
        model = Kosmos2Model(config).to(torch_device).eval()
        with torch.no_grad():
            result = model(input_ids, pixel_values, attention_mask)
        self.parent.assertEqual(
            result.logits_per_image.shape, (self.vision_model_tester.batch_size, self.text_model_tester.batch_size)
        )
        self.parent.assertEqual(
            result.logits_per_text.shape, (self.text_model_tester.batch_size, self.vision_model_tester.batch_size)
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, attention_mask, pixel_values = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "return_loss": True,
        }
        return config, inputs_dict


@require_torch
class Kosmos2ModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (Kosmos2Model,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"feature-extraction": Kosmos2Model, "image-to-text": Kosmos2ForConditionalGeneration}
        if is_torch_available()
        else {}
    )
    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False

    def setUp(self):
        self.model_tester = Kosmos2ModelTester(self)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_load_vision_text_config(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # Save Kosmos2Config and check if we can load Kosmos2VisionConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            vision_config = Kosmos2VisionConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.vision_config.to_dict(), vision_config.to_dict())

        # Save Kosmos2Config and check if we can load Kosmos2TextConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            text_config = Kosmos2TextConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.text_config.to_dict(), text_config.to_dict())

    @slow
    def test_model_from_pretrained(self):
        for model_name in Kosmos2_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = Kosmos2Model.from_pretrained(model_name)
            self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    url = "https://huggingface.co/hf-internal-testing/Kosmos2-test-image/resolve/main/demo.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@require_vision
@require_torch
@slow
class Kosmos2ModelIntegrationTest(unittest.TestCase):
    pass
