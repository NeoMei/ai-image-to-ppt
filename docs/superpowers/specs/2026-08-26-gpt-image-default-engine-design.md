# GPT Image 默认引擎设计

## 目标

为 `ai-image-to-ppt` 增加 OpenAI GPT Image 生成引擎，并让它成为程序和文档中的默认引擎。默认模型为 `gpt-image-2`，默认输出尺寸为 `2048x1152`，默认质量为 `medium`。现有 Gemini 和豆包引擎继续作为显式可选的备用引擎，已有直接调用方式保持兼容。另提供确定性的输入标准化工具，把任一引擎生成的 16:9 主图转换为 `image-to-editable-pptx` 当前要求的精确 `1280x720 PNG`。

## 非目标

- 不重构为通用 Provider 类体系。
- 不改变图片导出 PDF/PPTX 的逻辑。
- 不把视觉自检引擎从 Gemini 迁移到 OpenAI。
- 自动测试不发起真实付费图片生成请求。
- 不把所有生成与普通 PDF/PPTX 导出的全局分辨率降为 `1280x720`。
- 不在本插件中直接执行可编辑 PPTX 重建；这里只准备符合转换器契约的输入图片。

## 架构

### OpenAI 生成器

新增 `scripts/gen_slide_openai.py`，职责仅包括：

- 读取 OpenAI API 密钥和图像生成配置。
- 调用 OpenAI Images API。
- 执行有限重试并输出安全、简明的错误信息。
- 解码响应中的 Base64 图片数据并原子写入目标文件。
- 暴露与现有生成器一致的 `gen(prompt, out_path, retries=2) -> bool` 接口。

### 统一入口

新增 `scripts/gen_slide.py` 作为推荐入口：

- Python 接口为 `gen(prompt, out_path, engine="openai", retries=2) -> bool`。
- CLI 未指定 `--engine` 时选择 `openai`。
- CLI 支持 `--engine openai|gemini|doubao`。
- 只延迟导入实际选中的引擎，避免未配置备用引擎密钥时导入失败。
- 未知引擎直接返回清晰错误，不静默回退。

`gen_slide_gemini.py` 和 `gen_slide_doubao.py` 保持原有导入路径和直接调用方式。

### 可编辑转换输入标准化

新增 `scripts/prepare_editable_input.py`，作为所有生成引擎与 `image-to-editable-pptx` 之间的确定性交接层：

- 接受 Pillow 可解码的单张图片，输出精确 `1280x720` 的 PNG。
- 源图必须是严格 16:9，使用整数关系 `width * 9 == height * 16` 验证；非 16:9 图片直接拒绝，不静默裁剪或加边。
- 使用 LANCZOS 缩放并转为 RGB PNG，保留高分辨率源图不变。
- 输出路径必须以 `.png` 结尾，且不能与源路径相同。
- 已存在的输出文件默认拒绝覆盖。
- 通过同目录临时文件和不覆盖已有目标的原子发布方式写出结果，失败不留下半成品。
- 暴露 `prepare(input_path, output_path) -> bool`，并提供对应 CLI。

该工具不改变 OpenAI 默认的 `2048x1152` 主图参数，也不改变 `export_images.py` 面向普通 PDF/PPTX 的 `1920x1080` 输出。

## OpenAI 请求流程

### 凭证

按以下优先级读取密钥：

1. 环境变量 `OPENAI_API_KEY`。
2. 文件 `~/.secrets/openai_api_key`。

密钥在调用 `gen()` 时读取，不在模块导入时读取。缺少密钥时返回 `False` 并提示两种配置方式，任何日志均不得输出密钥内容。

### 默认参数与覆盖

请求发送到 `https://api.openai.com/v1/images/generations`，默认参数为：

- `model`: `gpt-image-2`
- `size`: `2048x1152`
- `quality`: `medium`

以下环境变量可以覆盖默认值：

- `OPENAI_IMAGE_MODEL`
- `OPENAI_IMAGE_SIZE`
- `OPENAI_IMAGE_QUALITY`

目标文件扩展名决定 `output_format`：

- `.jpg` 或 `.jpeg` → `jpeg`
- `.png` → `png`
- `.webp` → `webp`

不支持的扩展名在发送请求前失败，避免文件扩展名与真实编码不一致。

### 响应和文件写入

生成器读取 `data[0].b64_json` 并进行严格 Base64 解码。成功解码后先在目标目录创建临时文件，刷新并关闭后再用原子替换写入目标路径。请求、响应校验或解码失败时，不创建或覆盖最终文件，并清理临时文件。

## 错误处理与重试

- 网络错误、HTTP 429 和 HTTP 5xx 可以重试，最多执行 `retries + 1` 次请求。
- 其他 HTTP 4xx 不重试，包括鉴权错误、无效参数和内容策略拒绝。
- HTTP 错误优先提取 OpenAI JSON 响应中的 `error.message`，截断后输出，且不得包含请求头或密钥。
- 响应缺少图片数据、Base64 无效或本地写入失败时返回 `False`。
- 所有生成器继续遵守成功返回 `True`、失败返回 `False` 的既有约定。

## 文档更新

更新 `README.md` 和 `SKILL.md`：

- 将 GPT Image 2 标记为默认图像生成引擎。
- 将 Gemini nano banana 和豆包 Seedream 标记为备用引擎。
- 增加 `OPENAI_API_KEY` 与 `~/.secrets/openai_api_key` 配置说明。
- 默认单张与批量示例使用 `gen_slide.py`，不写 `--engine` 即走 OpenAI。
- 展示如何通过 CLI 和 Python 显式选择 Gemini 或豆包。
- 更新脚本表、成本说明和常见错误，避免继续把 Gemini 描述为首选。
- 保持 Gemini 视觉自检步骤不变，并明确它仍需 Gemini 密钥。
- 增加 `prepare_editable_input.py` 的单页标准化示例，并明确输出是可编辑转换器输入，不是可编辑 PPTX 本身。
- 说明 `image-to-editable-pptx` 当前只接受单张精确 `1280x720 PNG`，且后续转换仍需单独执行与验收。

## 测试策略

使用 Python 标准库 `unittest`，不新增测试依赖。测试必须隔离真实网络和真实密钥。

按测试先行顺序覆盖：

1. 统一入口未指定引擎时调用 OpenAI 生成器。
2. 显式选择 Gemini、豆包时调用对应生成器。
3. 未知引擎不会回退，并给出清晰错误。
4. `OPENAI_API_KEY` 优先于密钥文件；未设置时读取备用文件。
5. 默认请求包含 `gpt-image-2`、`2048x1152`、`medium` 和与扩展名一致的输出格式。
6. 环境变量可以覆盖模型、尺寸和质量。
7. 有效 `b64_json` 原子写入目标文件。
8. 无效响应或解码失败不会覆盖已有目标文件。
9. HTTP 429、HTTP 5xx 和网络错误重试；普通 HTTP 4xx 不重试。
10. `2048x1152` JPEG 等严格 16:9 输入可转换为真实编码的 `1280x720 PNG`。
11. 非 16:9 输入、非 `.png` 输出、源目标同路径和已存在目标均在写入前失败。
12. 标准化失败不改变源文件、已有目标文件，也不遗留临时文件。

实现完成后运行：

- 全量单元测试。
- `python3 -m compileall` Python 编译检查。
- 统一 CLI 的帮助和参数冒烟测试。
- `skill-creator` 提供的 `quick_validate.py` 技能结构校验。

真实 OpenAI API 冒烟不属于自动验收。若本机已配置密钥，仍需用户明确同意一次付费生成后才执行。

## 验收标准

- README 和 SKILL 均明确 GPT Image 2 是默认引擎。
- 默认 CLI 命令和默认 Python 调用均选择 OpenAI。
- 默认请求参数为 `gpt-image-2`、`2048x1152`、`medium`。
- 环境变量密钥优先，密钥文件可作为备用。
- Gemini 和豆包可通过统一入口显式选择，旧脚本仍可直接使用。
- 错误不会泄露密钥，也不会留下或覆盖损坏的目标文件。
- 任一引擎生成的严格 16:9 主图可通过独立命令得到精确 `1280x720 PNG`，并通过格式和像素尺寸检查。
- 主图生成默认值仍为 `2048x1152`，普通 PDF/PPTX 导出仍为 `1920x1080`；可编辑转换兼容不会降低现有默认清晰度。
- 文档不会把标准化 PNG 误称为可编辑 PPTX，并明确后续转换器的单页限制。
- 自动测试与静态验证全部通过，且不产生 API 费用。
