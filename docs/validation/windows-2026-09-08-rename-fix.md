# Windows 原生重命名修复与验证（2026-09-08）

## 结果

**PASS：修复后的 Windows 门禁通过。** 240 项测试中 239 项通过、0 失败、0 错误、1 项 POSIX 专属跳过；退出码 0。文档中的本地产物流原样执行通过。

本次解决了 [9635884 验证报告](windows-2026-09-07-9635884.md) 中的异常文件名、host 导入预期路径缺失和回滚预期路径缺失问题。之前的失败报告保留为修复前证据。

## 基线与修复

- 开发基线：`901a7c31ea6bf46b579fc27155965af8532b3caa`（含 9635884 候选代码和失败报告）。
- 本报告随修复代码和测试一起提交；被测代码为本次提交中的 `scripts/windows_image_output.py` 和 `tests/test_windows_image_output.py`。
- `_NativeApi.rename` 原先仅分配路径实际字节空间，没有在 UTF-16 字符串后预留 NUL WCHAR，可能使后续内存成为文件名后缀。
- 现在额外分配一个 `ctypes.sizeof(WCHAR)` 的零初始化空间；FileNameLength 仍是路径实际字节数，不含终止字符。
- 句柄绑定重命名、禁止覆盖、共享冲突、身份检查及恢复语义保持原样。

## 回归测试先失败，再修复

新增两项 Windows 回归测试：

1. 在 Win32 调用边界检查真实构造的缓冲区：ASCII、中文/空格、emoji 路径的 UTF-16 内容正确、终止 WCHAR 为零、FileNameLength 不包含终止字符。只拦截系统调用以安全检查 ABI 参数。
2. 使用真实 Win32 文件操作和 host importer，在 6 种目录长度及中文、空格、希腊字母、emoji 组合下检查 raw/master 文件名集合、输出路径和字节内容，禁止意外后缀。

修改产品代码前，两项新测试均失败：缓冲区检查的 3 个子用例发现 `b'' != b'\x00\x00'`，真实导入检查的一个子用例返回 `local_failure`。终端摘要 `Ran 2 tests ... FAILED (failures=4)` 中的 4 是失败子用例数量。

应用修复后，没有进程内替换或诊断补丁，直接运行产品源码：

| 验证 | 结果 |
| --- | --- |
| Windows 原生发布模块 | 26/26 通过；0.572 秒；退出码 0 |
| README 指定 Windows 门禁 | 240 项，239 通过、1 跳过；24.474 秒；退出码 0 |
| Skill 校验 | `Skill is valid!` |
| 依赖一致性 | `No broken requirements found.` |
| 文档本地产物流 | PASS：导入、raw 保留、拒绝覆盖、显式替换、editable、PPTX/PDF |

唯一跳过项为 `test_capability_manifest.CapabilityManifestTests.test_fifo_script_is_rejected_as_non_regular`，原因 `requires POSIX FIFO support`。原生 Win32 测试和符号链接拒绝检查均执行，没有因缺少 Windows 权限跳过。

原先报错的 `test_local_file_is_validated_and_published_inside_workspace` 与 `test_host_pair_rolls_back_after_master_checkpoint_failure` 在完整 Windows 门禁中均通过。

## 环境与产物

- 日期：2026-09-08，Asia/Shanghai（UTC+08:00）。
- Windows 11，`Windows-11-10.0.26200-SP0`；C:、D: 为本地固定 NTFS 卷。
- CPython 3.14.3，AMD64。
- 使用上一轮新建的隔离虚拟环境：`.worktrees/windows-9635884/.venv/Scripts/python.exe`；本轮工作目录为主仓库根目录，执行的是已修复的主仓库源码。
- 依赖：Pillow 12.3.0、python-pptx 1.0.2、PyYAML 6.0.3、lxml 6.1.3、XlsxWriter 3.2.9、typing-extensions 4.16.0。
- 本地产物目录：`C:\Users\1\AppData\Local\Temp\ai-image-windows-7i56sdau\中文 空格`。

该产物流验证 1672×941 原图保留、1664×936 主图、精确 1280×720 editable PNG、单页 16:9 PPTX 用 python-pptx 重开，以及非空 PDF。

## 日志与复现

[回归修复前日志](evidence-rename-fix-2026-09-08/regression-red.log)、[原生模块通过日志](evidence-rename-fix-2026-09-08/native-green.log)、[完整 Windows 门禁日志](evidence-rename-fix-2026-09-08/windows-green.log)、[文档产物流通过日志](evidence-rename-fix-2026-09-08/artifact-green.log)。故障注入用例的预期 `ERR:`/恢复警告不表示测试失败。

从安装了 requirements-dev.txt 依赖的 Windows 环境，在本修复版本的仓库根目录执行：

```powershell
python scripts/validate_skill.py .
if ($LASTEXITCODE -ne 0) { throw 'Skill validation failed' }
python scripts/run_windows_tests.py
if ($LASTEXITCODE -ne 0) { throw 'Windows tests failed' }
```

随后原样执行 [Windows 验证指南](../windows-validation.md) 的 Local artifact flow 代码块。

Git HTTPS 曾连接失败，本轮通过 GitHub Git API 补齐本地缺少的原始提交、树和 blob 对象，逐对象核验 SHA，并经 `git fsck --connectivity-only` 检查后正常快进到远端基线。没有生成替代基线或截断历史；原本地报告副本核对与远端一致后备份在 `out/published-report-backup-901a7c3`。

本次未运行真实付费 provider API、宿主图片生成或 Office/WPS 桌面验收；上述测试证明本地 Windows 发布与导出链路。没有验证物理断电、UNC/网络盘、云同步目录或其他操作系统。
