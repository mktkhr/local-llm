# 出典

ComfyUI 公式ワークフローテンプレート(Comfy-Org/workflow_templates、MIT License)の UI 形式 JSON をそのまま同梱している。

- 取得元: Docker イメージ `yanwk/comfyui-boot:cu130-slim-v2` 内の pip パッケージ `comfyui-workflow-templates` 0.11.97
  (`/usr/local/lib/python3.13/site-packages/comfyui_workflow_templates_json/templates/`)
- 取得日: 2026-11-03
- 用途: `tpl2api.py` で API 形式に変換し、`comfy_api.py` / `run_matrix.py` から実行する。公式の推奨設定
  (サンプラー・ステップ数・CFG・shift 等)をそのまま使うため、手書きグラフではなくテンプレートを基準にする。
