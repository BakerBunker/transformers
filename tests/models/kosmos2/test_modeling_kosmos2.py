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


import copy
import inspect
import numpy as np
import os
import requests
import tempfile
import unittest


from transformers.models.kosmos2.configuration_kosmos2 import Kosmos2TextConfig, Kosmos2VisionConfig
from transformers import AutoProcessor, AutoModelForVision2Seq, Kosmos2Config
from transformers.testing_utils import require_torch, require_vision, slow, torch_device
from transformers.utils import is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    _config_zero_init,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)


if is_torch_available():
    import torch

    from transformers import Kosmos2Model, Kosmos2ForConditionalGeneration
    from transformers.models.kosmos2.modeling_kosmos2 import KOSMOS2_PRETRAINED_MODEL_ARCHIVE_LIST


if is_vision_available():
    from PIL import Image


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

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, input_mask = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


class Kosmos2ModelTester:
    def __init__(self, parent, text_kwargs=None, vision_kwargs=None, latent_query_num=3, is_training=True):
        if text_kwargs is None:
            text_kwargs = {}
        if vision_kwargs is None:
            vision_kwargs = {}

        self.parent = parent
        self.text_model_tester = Kosmos2TextModelTester(parent, **text_kwargs)
        self.vision_model_tester = Kosmos2VisionModelTester(parent, **vision_kwargs)
        self.latent_query_num = latent_query_num
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        text_config, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()
        vision_config, pixel_values = self.vision_model_tester.prepare_config_and_inputs()

        # build `image_features_mask`
        image_features_mask = torch.zeros_like(input_ids)
        image_features_mask[:, 1: 1 + self.latent_query_num:] = 1

        config = self.get_config()

        return config, input_ids, attention_mask, image_features_mask, pixel_values

    def get_config(self):
        return Kosmos2Config(
            self.text_model_tester.get_config().to_dict(),
            self.vision_model_tester.get_config().to_dict(),
            latent_query_num=self.latent_query_num,
        )

    def create_and_check_model(self, config, input_ids, attention_mask, image_features_mask, pixel_values):
        model = Kosmos2Model(config).to(torch_device).eval()
        with torch.no_grad():
            result = model(pixel_values, input_ids, image_features_mask, attention_mask)
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.text_model_tester.batch_size, self.text_model_tester.seq_length, self.text_model_tester.hidden_size)
        )
        self.parent.assertEqual(
            result.image_features.shape, (self.text_model_tester.batch_size, self.latent_query_num, self.text_model_tester.hidden_size)
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, attention_mask, image_features_mask, pixel_values = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_features_mask": image_features_mask,
            "pixel_values": pixel_values,
        }
        return config, inputs_dict


@require_torch
class Kosmos2ModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (Kosmos2Model, Kosmos2ForConditionalGeneration) if is_torch_available() else ()
    all_generative_model_classes = (Kosmos2ForConditionalGeneration,) if is_torch_available() else ()
    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = copy.deepcopy(inputs_dict)

        if return_labels:
            if model_class.__name__ == "Kosmos2ForConditionalGeneration":
                inputs_dict["labels"] = torch.zeros(
                    (self.model_tester.text_model_tester.batch_size, self.model_tester.text_model_tester.seq_length), dtype=torch.long, device=torch_device
                )

        return inputs_dict

    def setUp(self):
        self.model_tester = Kosmos2ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Kosmos2Config, hidden_size=37)

    # TODO: Remove this
    @unittest.skip("Not applicable.")
    def test_config(self):
        self.config_tester.run_common_tests()

    # TODO: Implement `_init_weights` and remove the skip.
    @unittest.skip("`_init_weights` is not yet implemented for `KOSMOS-2` model.")
    def test_initialization(self):
        super().test_initialization()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    # over... from common
    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states

            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", self.model_tester.text_model_tester.num_hidden_layers + 1
            )
            self.assertEqual(len(hidden_states), expected_num_layers)

            seq_length = self.model_tester.text_model_tester.seq_length

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [seq_length, self.model_tester.text_model_tester.hidden_size],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    def test_tie_model_weights(self):
        if not self.test_torchscript:
            return

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_same_values(layer_1, layer_2):
            equal = True
            for p1, p2 in zip(layer_1.weight, layer_2.weight):
                if p1.data.ne(p2.data).sum() > 0:
                    equal = False
            return equal

        for model_class in self.all_model_classes:
            config.torchscript = True
            model_not_tied = model_class(config)
            if model_not_tied.get_output_embeddings() is None:
                continue

            config_tied = copy.deepcopy(config)
            config_tied.torchscript = False
            model_tied = model_class(config_tied)
            params_tied = list(model_tied.parameters())
            # Check that the embedding layer and decoding layer are the same in size and in value
            # self.assertTrue(check_same_values(embeddings, decoding))

            # # Check that after modification, they remain the same.
            # embeddings.weight.data.div_(2)
            # # Check that the embedding layer and decoding layer are the same in size and in value
            # self.assertTrue(embeddings.weight.shape, decoding.weight.shape)
            # self.assertTrue(check_same_values(embeddings, decoding))

            # # Check that after modification, they remain the same.
            # decoding.weight.data.div_(4)
            # # Check that the embedding layer and decoding layer are the same in size and in value
            # self.assertTrue(embeddings.weight.shape, decoding.weight.shape)
            # self.assertTrue(check_same_values(embeddings, decoding))

            # Check that after resize they remain tied.
            model_tied.resize_token_embeddings(config.text_config.vocab_size + 10)
            params_tied_2 = list(model_tied.parameters())
            self.assertEqual(len(params_tied_2), len(params_tied))

            # decoding.weight.data.mul_(20)
            # # Check that the embedding layer and decoding layer are the same in size and in value
            # self.assertTrue(model.transformer.wte.weight.shape, model.lm_head.weight.shape)
            # self.assertTrue(check_same_values(model.transformer.wte, model.lm_head))

    @slow
    def test_model_from_pretrained(self):
        for model_name in KOSMOS2_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = Kosmos2Model.from_pretrained(model_name)
            self.assertIsNotNone(model)

    def _create_and_check_torchscript(self, config, inputs_dict):
        if not self.test_torchscript:
            return

        configs_no_init = _config_zero_init(config)  # To be sure we have no Nan
        configs_no_init.torchscript = True
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            model.to(torch_device)
            model.eval()
            inputs = self._prepare_for_class(inputs_dict, model_class)

            main_input_name = model_class.main_input_name

            try:
                main_input = inputs[main_input_name]
                model(main_input, inputs['input_ids'], inputs['image_features_mask'])
                traced_model = torch.jit.trace(model, (main_input, inputs['input_ids'], inputs['image_features_mask']))
            except RuntimeError:
                self.fail("Couldn't trace module.")

            with tempfile.TemporaryDirectory() as tmp_dir_name:
                pt_file_name = os.path.join(tmp_dir_name, "traced_model.pt")

                try:
                    torch.jit.save(traced_model, pt_file_name)
                except Exception:
                    self.fail("Couldn't save module.")

                try:
                    loaded_model = torch.jit.load(pt_file_name)
                except Exception:
                    self.fail("Couldn't load module.")

            model.to(torch_device)
            model.eval()

            loaded_model.to(torch_device)
            loaded_model.eval()

            model_state_dict = model.state_dict()
            loaded_model_state_dict = loaded_model.state_dict()

            non_persistent_buffers = {}
            for key in loaded_model_state_dict.keys():
                if key not in model_state_dict.keys():
                    non_persistent_buffers[key] = loaded_model_state_dict[key]

            loaded_model_state_dict = {
                key: value for key, value in loaded_model_state_dict.items() if key not in non_persistent_buffers
            }

            self.assertEqual(set(model_state_dict.keys()), set(loaded_model_state_dict.keys()))

            model_buffers = list(model.buffers())
            for non_persistent_buffer in non_persistent_buffers.values():
                found_buffer = False
                for i, model_buffer in enumerate(model_buffers):
                    if torch.equal(non_persistent_buffer, model_buffer):
                        found_buffer = True
                        break

                self.assertTrue(found_buffer)
                model_buffers.pop(i)

            models_equal = True
            for layer_name, p1 in model_state_dict.items():
                if layer_name in loaded_model_state_dict:
                    p2 = loaded_model_state_dict[layer_name]
                    if p1.data.ne(p2.data).sum() > 0:
                        models_equal = False

            self.assertTrue(models_equal)

            # Avoid memory leak. Without this, each call increase RAM usage by ~20MB.
            # (Even with this call, there are still memory leak by ~0.04MB)
            self.clear_torch_jit_class_registry()


# We will verify our results on an image of cute cats
def prepare_img():
    url = "https://huggingface.co/hf-internal-testing/Kosmos2-test-image/resolve/main/demo.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@require_vision
@require_torch
@slow
class Kosmos2ModelIntegrationTest(unittest.TestCase):

    def test_snowman_image_captioning(self):

        url = "https://huggingface.co/ydshieh/kosmos-2-patch14-224/resolve/main/snowman.png"

        image = Image.open(requests.get(url, stream=True).raw)
        image.save("new_image.jpg")
        image = Image.open("new_image.jpg")

        model = AutoModelForVision2Seq.from_pretrained("ydshieh/kosmos-2-patch14-224", trust_remote_code=True)
        processor = AutoProcessor.from_pretrained("ydshieh/kosmos-2-patch14-224", trust_remote_code=True)

        def run_example(prompt):

            inputs = processor(text=prompt, images=image, return_tensors="pt")

            generation_outputs = model.generate(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"][:, :-1],
                attention_mask=inputs["attention_mask"][:, :-1],
                img_features=None,
                img_attn_mask=inputs["img_attn_mask"][:, :-1],
                use_cache=True,
                max_new_tokens=64,
                output_scores=True,
                return_dict_in_generate=True,
            )

            scores = generation_outputs.scores
            generated_ids = generation_outputs.sequences
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            # Specify `cleanup_and_extract=False` in order to see the raw model generation.
            processed_text = processor.post_process_generation(generated_text, cleanup_and_extract=False)
            # By default, the generated  text is cleanup and the entities are extracted.
            final_text, entities = processor.post_process_generation(generated_text)

            return scores, generated_ids, generated_text, processed_text, final_text, entities

        prompt = "<grounding>An image of"
        scores, generated_ids, generated_text, processed_text, final_text, entities = run_example(prompt)

        assert torch.concat(scores[:3])[:3, :3].numpy().tolist() == [
            [-2.266045093536377, -5.1130499839782715, 4.401935577392578],
            [-1.5672576427459717, -5.007407188415527, 4.364485740661621],
            [-2.1470165252685547, -4.96630334854126, 4.592554569244385],
        ]
        assert torch.concat(scores[-3:])[-3:, -3:].numpy().tolist() == [
            [2.9916181564331055, 2.4818153381347656, 4.646588325500488],
            [-2.8381104469299316, -2.9687182903289795, -2.6926774978637695],
            [-2.890916347503662, -3.2228598594665527, -1.7056852579116821],
        ]

        assert generated_ids.numpy().tolist() == [
            [
                0, 64003, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 64004, 64012, 712, 1648, 9, 64007, 10, 43867, 64008,
                64009, 64057, 64876, 64010, 5950, 597, 32, 64007, 10, 646, 64008, 64009, 64018, 64924, 64010, 4, 2
            ]
        ]

        assert processed_text == (
            "<grounding> An image of<phrase> a snowman</phrase><object><patch_index_0044><patch_index_0863></object> "
            "warming himself by<phrase> a fire</phrase><object><patch_index_0005><patch_index_0911></object>."
        )

        assert final_text == "An image of a snowman warming himself by a fire."
        assert entities == [
            ('a snowman', (12, 21), [(0.390625, 0.046875, 0.984375, 0.828125)]),
            ('a fire', (41, 47), [(0.171875, 0.015625, 0.484375, 0.890625)]),
        ]

        prompt = "<grounding>Describe this image in detail:"
        scores, generated_ids, generated_text, processed_text, final_text, entities = run_example(prompt)

        assert torch.concat(scores[:3])[:3, :3].numpy().tolist() == [
            [-1.17293381690979, -4.673310279846191, 7.595856666564941],
            [-0.9093594551086426, -4.578374862670898, 5.963612079620361],
            [2.4521265029907227, -4.090599536895752, 8.738670349121094],
        ]
        assert torch.concat(scores[-3:])[-3:, -3:].numpy().tolist() == [
            [-1.6924376487731934, -1.5812195539474487, -1.222994327545166],
            [-1.2709499597549438, -1.478440284729004, -0.8769001960754395],
            [6.566478252410889, 5.109105110168457, 3.3479409217834473],
        ]

        assert generated_ids.numpy().tolist() == [
            [
                0, 64003, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 64004, 64012, 34645, 247, 38, 1648, 12, 3391, 55,
                24, 1648, 1338, 10, 43867, 1280, 32, 64007, 10, 30879, 64008, 64009, 64018, 65020, 64010, 12, 5, 1842,
                4, 71, 17, 1679, 64007, 10, 3958, 64008, 64009, 64061, 64263, 64010, 6, 64007, 15719, 64008, 64009,
                64253, 64617, 64010, 6, 8, 64007, 9626, 64008, 64009, 64413, 64545, 64010, 6, 23, 64007, 10, 4363,
                64008, 64009, 64623, 64885, 64010, 2255, 8, 64007, 10, 3486, 64008, 64009
            ]
        ]

        assert processed_text == (
            "<grounding> Describe this image in detail: The image features a snowman sitting by<phrase> a campfire"
            "</phrase><object><patch_index_0005><patch_index_1007></object> in the snow. He is wearing<phrase> a hat"
            "</phrase><object><patch_index_0048><patch_index_0250></object>,<phrase> scarf</phrase><object>"
            "<patch_index_0240><patch_index_0604></object>, and<phrase> gloves</phrase><object><patch_index_0400>"
            "<patch_index_0532></object>, with<phrase> a pot</phrase><object><patch_index_0610><patch_index_0872>"
            "</object> nearby and<phrase> a cup</phrase><object>"
        )
        assert final_text == (
            "Describe this image in detail: The image features a snowman sitting by a campfire in the snow. He is "
            "wearing a hat, scarf, and gloves, with a pot nearby and a cup"
        )
        assert entities == [
            ('a campfire', (71, 81), [(0.171875, 0.015625, 0.484375, 0.984375)]),
            ('a hat', (109, 114), [(0.515625, 0.046875, 0.828125, 0.234375)]),
            ('scarf', (116, 121), [(0.515625, 0.234375, 0.890625, 0.578125)]),
            ('gloves', (127, 133), [(0.515625, 0.390625, 0.640625, 0.515625)]),
            ('a pot', (140, 145), [(0.078125, 0.609375, 0.265625, 0.859375)]),
        ]
