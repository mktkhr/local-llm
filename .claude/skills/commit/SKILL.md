---
name: commit
description: '変更を適切な粒度に分割してコミットする。"コミット", "commit", "変更をコミット" 等で発動。'
---
# スマートコミット（設計書リポジトリ専用）

変更を関心事ごとに分割し、規約に従ったコミットメッセージで個別にコミットする。

## コミットメッセージ規約

```
{type}:{emoji} {対象の説明}
```

- subject 1行のみ。body（詳細説明）は書かない
- 日本語で記載
- "Generated with Claude Code" / "Co-Authored-By: Claude" を含めない

### このリポジトリで使用する Type + Emoji

| Type     | 用途                                      | Emoji |
| -------- | ----------------------------------------- | ----- |
| docs     | 設計書・テンプレート・スキル・ADR・README | 📝    |
| refactor | リネーム・構造変更（内容変更なし）        | ♻️    |
| chore    | 設定・hooks・prettier・gitignore 等       | 🔧    |
| fix      | 誤記修正・パス修正・不整合修正            | 🐛    |

使用頻度が低い（必要に応じて）:

| Type   | 用途             | Emoji |
| ------ | ---------------- | ----- |
| add    | 新規ファイル追加 | ✨    |
| remove | ファイル削除     | 🗑️    |

このリポジトリにコードはないため、`test` / `style` / `security` は使わない。

## 手順

0. **品質チェック&フォーマット (必須・最初に実行)**
   - `make format` を実行して prettier で対象ファイルを整形する
   - 整形による差分は git status に追加で現れるため、後続の関心事ごとのコミットに自然に含める
     (整形差分のみを別コミットに分離する必要はない)
   - prettier 実行が失敗した場合(構文エラーなど)はコミットを中断し、原因を報告する
1. `git status` と `git diff` で全変更を把握する
2. `git log --oneline -10` で直近のコミットメッセージパターンを確認する
3. 変更を関心事ごとにグループ分けする
4. グループごとに:
   - 対象ファイルのみ `git add` でステージング
   - `git diff --cached` で差分を確認
   - コミットメッセージでコミット
5. 全コミット完了後、`git log --oneline` で結果を表示する

## 分割基準

| 関心事             | 例                                             |
| ------------------ | ---------------------------------------------- |
| 設計書本体         | `specs/{機能名}/index.md`                      |
| 実装プロンプト     | `specs/{機能名}/prompts/`                      |
| テンプレート       | `templates/`                                   |
| スキル(個別)     | `.claude/skills/{スキル名}/`                   |
| スキル(リネーム) | 旧スキル削除 + 新スキル追加をセットで          |
| ルール             | `.claude/rules/`                               |
| エージェント       | `.claude/agents/`                              |
| hooks / settings   | `.claude/hooks/` + `.claude/settings.json`     |
| フロントエンド設定 | `apps/frontend/.claude/`(別リポジトリの場合) |
| ADR                | `adr/`                                         |
| README / CLAUDE.md | 各ファイル                                     |
| prettier / vscode  | `.prettierrc` + `.vscode/`                     |

ユーザーから「めっちゃ細かく」等の指示があればそれに従う。

## 禁止事項

- `git add -A` は使わない
- 1つのコミットに全変更をまとめない
- .env, credentials, トークン等を含むファイルはコミットしない
- `.ai-out/` 配下はコミットしない(.gitignore で除外済み)
