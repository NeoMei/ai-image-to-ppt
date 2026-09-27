import base64
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import gen_slide
import gen_slide_doubao
from generation_result import GenerationStatus

class Response(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *args): self.close()

class ReferenceImageTests(unittest.TestCase):
    def test_doubao_rejects_webp_output_before_credentials_or_network(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(gen_slide_doubao, '_load_api_key') as key, mock.patch.object(gen_slide_doubao.urllib.request, 'urlopen') as network:
            result = gen_slide.generate_result('prompt', str(Path(directory) / 'out.webp'), engine='doubao', retries=0)
        self.assertEqual(result.status, GenerationStatus.INVALID_INPUT)
        key.assert_not_called()
        network.assert_not_called()

    def test_reference_reaches_doubao_request_without_changing_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            ref = Path(directory) / 'reference.png'
            Image.new('RGB', (160, 90), 'teal').save(ref)
            response = Response(json.dumps({'data': [{'url': 'https://example.com/result.png'}]}).encode())
            with mock.patch.object(gen_slide_doubao, '_load_api_key', return_value='test-key'), mock.patch.object(gen_slide_doubao.urllib.request, 'urlopen', return_value=response) as network, mock.patch.object(gen_slide_doubao, '_download_image_result', return_value=gen_slide_doubao._result(GenerationStatus.SUCCESS, 'ok', str(Path.cwd() / 'out.png'))):
                result = gen_slide.generate_result('exact prompt', str(Path(directory) / 'out.png'), engine='doubao', retries=0, reference_images=[str(ref)])
            self.assertTrue(result.ok)
            body = json.loads(network.call_args.args[0].data)
            self.assertEqual(body['prompt'], 'exact prompt')
            self.assertEqual(body['image'], ['data:image/png;base64,' + base64.b64encode(ref.read_bytes()).decode()])
            self.assertNotIn('sequential_image_generation', body)

    def test_bad_reference_is_rejected_before_credentials_or_network(self):
        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / 'bad.png'
            bad.write_text('not an image')
            with mock.patch.object(gen_slide_doubao, '_load_api_key') as key, mock.patch.object(gen_slide_doubao.urllib.request, 'urlopen') as network:
                result = gen_slide_doubao.generate_result('prompt', str(Path(directory) / 'out.png'), reference_images=[str(bad)])
            self.assertEqual(result.status, GenerationStatus.INVALID_INPUT)
            key.assert_not_called()
            network.assert_not_called()

    def test_unsupported_adapter_cannot_drop_references(self):
        for engine in ['openai', 'gemini']:
            with mock.patch.object(gen_slide.importlib, 'import_module') as provider:
                result = gen_slide.generate_result('prompt', 'out.png', engine=engine, reference_images=['reference.png'])
            self.assertEqual(result.status, GenerationStatus.UNAVAILABLE)
            provider.assert_not_called()

    def test_cli_forwards_repeatable_reference(self):
        with mock.patch.object(gen_slide, 'generate_result', return_value=gen_slide_doubao._result(GenerationStatus.SUCCESS, 'ok', str(Path.cwd() / 'out.png'))) as generate:
            gen_slide.main(['out.png', 'prompt', '--engine', 'doubao', '--reference-image', 'one.png', '--reference-image', 'two.png', '--json'])
        self.assertEqual(generate.call_args.kwargs['reference_images'], ['one.png', 'two.png'])

    def test_empty_reference_list_keeps_legacy_provider_signature(self):
        from types import SimpleNamespace
        expected = gen_slide._result(GenerationStatus.UNAVAILABLE, 'openai', 'fixture')
        provider = SimpleNamespace(generate_result=mock.Mock(return_value=expected))
        with mock.patch.object(gen_slide.importlib, 'import_module', return_value=provider):
            result = gen_slide.generate_result('prompt', 'out.png', reference_images=[])
        self.assertEqual(result, expected)
        self.assertNotIn('reference_images', provider.generate_result.call_args.kwargs)

    def test_reference_limits_and_types_fail_without_network(self):
        for refs in ['file.png', 0, ['file.png'] * 11]:
            with mock.patch.object(gen_slide_doubao.urllib.request, 'urlopen') as network:
                result = gen_slide.generate_result('prompt', 'out.png', engine='doubao', reference_images=refs)
            self.assertEqual(result.status, GenerationStatus.INVALID_INPUT)
            network.assert_not_called()

    def test_pro_and_flash_omit_unsupported_group_parameter(self):
        for model in [None, 'doubao-seedream-5-0-flash-260915', 'doubao-seedream-5-0-pro-260628']:
            with tempfile.TemporaryDirectory() as directory, mock.patch.dict('os.environ', {}, clear=True):
                if model:
                    import os
                    os.environ['DOUBAO_IMAGE_MODEL'] = model
                response = Response(json.dumps({'data': [{'url': 'https://example.com/result.png'}]}).encode())
                with mock.patch.object(gen_slide_doubao, '_load_api_key', return_value='test-key'), mock.patch.object(gen_slide_doubao.urllib.request, 'urlopen', return_value=response) as network, mock.patch.object(gen_slide_doubao, '_download_image_result', return_value=gen_slide_doubao._result(GenerationStatus.SUCCESS, 'ok', str(Path(directory) / 'out.png'))):
                    result = gen_slide.generate_result('exact prompt', str(Path(directory) / 'out.png'), engine='doubao', retries=0)
                self.assertTrue(result.ok)
                body = json.loads(network.call_args.args[0].data)
                self.assertEqual(body['model'], model or 'doubao-seedream-5-0-pro-260628')
                self.assertNotIn('sequential_image_generation', body)

    def test_lite_override_retains_legacy_request(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict('os.environ', {'DOUBAO_IMAGE_MODEL': 'doubao-seedream-5-0-260128'}):
            response = Response(json.dumps({'data': [{'url': 'https://example.com/result.png'}]}).encode())
            with mock.patch.object(gen_slide_doubao, '_load_api_key', return_value='test-key'), mock.patch.object(gen_slide_doubao.urllib.request, 'urlopen', return_value=response) as network, mock.patch.object(gen_slide_doubao, '_download_image_result', return_value=gen_slide_doubao._result(GenerationStatus.SUCCESS, 'ok', str(Path(directory) / 'out.png'))):
                gen_slide.generate_result('exact prompt', str(Path(directory) / 'out.png'), engine='doubao', retries=0)
            self.assertEqual(json.loads(network.call_args.args[0].data)['sequential_image_generation'], 'disabled')
