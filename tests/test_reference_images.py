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
import gen_slide_gemini
import gen_slide_openai
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

    def test_openai_sends_references_to_edits_with_current_model(self):
        from email import policy
        from email.parser import BytesParser

        with tempfile.TemporaryDirectory() as directory:
            ref = Path(directory) / 'reference.png'
            Image.new('RGB', (160, 90), 'teal').save(ref)
            response = Response(json.dumps({'data': [{
                'b64_json': base64.b64encode(ref.read_bytes()).decode(),
            }]}).encode())
            with mock.patch.object(gen_slide_openai, '_load_api_key', return_value='test-key'), mock.patch.object(gen_slide_openai.urllib.request, 'urlopen', return_value=response) as network:
                result = gen_slide.generate_result('确切的 prompt', str(Path(directory) / 'out.png'), retries=0, reference_images=[str(ref), str(ref)])
            self.assertTrue(result.ok)
            network.assert_called_once()
            request = network.call_args.args[0]
            self.assertEqual(request.full_url, 'https://api.openai.com/v1/images/edits')
            mime = BytesParser(policy=policy.default).parsebytes(
                f"Content-Type: {request.get_header('Content-type')}\r\n\r\n".encode() + request.data
            )
            fields, images = {}, []
            for part in mime.iter_parts():
                name = part.get_param('name', header='content-disposition')
                if name == 'image[]':
                    self.assertEqual(part.get_content_type(), 'image/png')
                    images.append(part.get_payload(decode=True))
                else:
                    fields[name] = part.get_payload(decode=True).decode()
            self.assertEqual(images, [ref.read_bytes()] * 2)
            self.assertEqual(fields, {
                'model': 'gpt-image-2.6-sunburst',
                'prompt': '确切的 prompt',
                'size': '2048x1152',
                'quality': 'medium',
                'output_format': 'png',
            })

    def test_gemini_sends_references_and_normalizes_documented_2k_size(self):
        with tempfile.TemporaryDirectory() as directory:
            ref = Path(directory) / 'reference.png'
            out = Path(directory) / 'out.png'
            Image.new('RGB', (160, 90), 'teal').save(ref)
            canvas = io.BytesIO()
            image = Image.new('RGB', (2752, 1536), 'white')
            image.putpixel((0, 0), (255, 0, 0))
            image.putpixel((2751, 1535), (0, 0, 255))
            image.save(canvas, format='PNG')
            response = Response(json.dumps({'candidates': [{'content': {'parts': [
                {'thought': True, 'inlineData': {'mimeType': 'image/png', 'data': 'interim-image'}},
                {'inlineData': {'mimeType': 'image/png', 'data': base64.b64encode(canvas.getvalue()).decode()}},
            ]}}]}).encode())
            with mock.patch.object(gen_slide_gemini, '_load_api_key', return_value='test-key'), mock.patch.object(gen_slide_gemini.urllib.request, 'urlopen', return_value=response) as network:
                result = gen_slide.generate_result('exact prompt', str(out), engine='gemini', retries=0, reference_images=[str(ref)])
            self.assertTrue(result.ok)
            network.assert_called_once()
            request = network.call_args.args[0]
            self.assertIn('/v1/models/gemini-nano-banana-2.1:generateContent', request.full_url)
            body = json.loads(request.data)
            self.assertEqual(body['contents'][0]['parts'], [
                {'text': 'exact prompt'},
                {'inlineData': {'mimeType': 'image/png', 'data': base64.b64encode(ref.read_bytes()).decode()}},
            ])
            self.assertEqual(body['generationConfig']['responseFormat']['image'], {
                'aspectRatio': 'ASPECT_RATIO_SIXTEEN_BY_NINE', 'imageSize': 'IMAGE_SIZE_TWO_K',
            })
            with Image.open(out) as master:
                self.assertEqual(master.size, (2752, 1548))
                self.assertEqual(master.getpixel((0, 6)), (255, 0, 0))
                self.assertEqual(master.getpixel((2751, 1541)), (0, 0, 255))

    def test_openai_and_gemini_reject_bad_references_before_network(self):
        for engine, adapter in [('openai', gen_slide_openai), ('gemini', gen_slide_gemini)]:
            with self.subTest(engine=engine), mock.patch.object(adapter, '_load_api_key') as key, mock.patch.object(adapter.urllib.request, 'urlopen') as network:
                result = gen_slide.generate_result('prompt', 'out.png', engine=engine, reference_images=['not-a-local-file.png'])
                self.assertEqual(result.status, GenerationStatus.INVALID_INPUT)
                key.assert_not_called()
                network.assert_not_called()

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
        for model in [None, 'doubao-seedream-5-0-flash-260915', 'doubao-seedream-5-0-pro-260628', 'ep-custom-endpoint']:
            with tempfile.TemporaryDirectory() as directory, mock.patch.dict('os.environ', {}, clear=True):
                if model:
                    import os
                    os.environ['DOUBAO_IMAGE_MODEL'] = model
                response = Response(json.dumps({'data': [{'url': 'https://example.com/result.png'}]}).encode())
                with mock.patch.object(gen_slide_doubao, '_load_api_key', return_value='test-key'), mock.patch.object(gen_slide_doubao.urllib.request, 'urlopen', return_value=response) as network, mock.patch.object(gen_slide_doubao, '_download_image_result', return_value=gen_slide_doubao._result(GenerationStatus.SUCCESS, 'ok', str(Path(directory) / 'out.png'))):
                    result = gen_slide.generate_result('exact prompt', str(Path(directory) / 'out.png'), engine='doubao', retries=0)
                self.assertTrue(result.ok)
                body = json.loads(network.call_args.args[0].data)
                self.assertEqual(body['model'], model or 'doubao-seedream-5-5-261028')
                self.assertNotIn('sequential_image_generation', body)

    def test_lite_override_retains_legacy_request(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict('os.environ', {'DOUBAO_IMAGE_MODEL': 'doubao-seedream-5-0-260128'}):
            response = Response(json.dumps({'data': [{'url': 'https://example.com/result.png'}]}).encode())
            with mock.patch.object(gen_slide_doubao, '_load_api_key', return_value='test-key'), mock.patch.object(gen_slide_doubao.urllib.request, 'urlopen', return_value=response) as network, mock.patch.object(gen_slide_doubao, '_download_image_result', return_value=gen_slide_doubao._result(GenerationStatus.SUCCESS, 'ok', str(Path(directory) / 'out.png'))):
                gen_slide.generate_result('exact prompt', str(Path(directory) / 'out.png'), engine='doubao', retries=0)
            self.assertEqual(json.loads(network.call_args.args[0].data)['sequential_image_generation'], 'disabled')
