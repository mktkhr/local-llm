.PHONY: help rulesync-import rulesync-generate rulesync format format-check setup doctor

RULESYNC_VERSION ?= 8.3.0
RULESYNC_FEATURES ?= rules,ignore,mcp,subagents,commands,skills,hooks,permissions
RULESYNC_TARGETS ?= claudecode,codexcli,cline

# prettier: メジャー固定 + マイナー/パッチは最新を追従
PRETTIER_VERSION ?= 3

help: ## ヘルプ
	@awk 'BEGIN {FS = ":.*##"} /^([a-zA-Z0-9_-]+):.*##/ { printf "\033[36m%-18s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

rulesync-import: ## rulesync: claudecode からルールを取り込み
	npx -y rulesync@$(RULESYNC_VERSION) import --targets claudecode --features $(RULESYNC_FEATURES)

rulesync-generate: ## rulesync: claudecode,codexcli,cline 向けに生成
	npx -y rulesync@$(RULESYNC_VERSION) generate --targets $(RULESYNC_TARGETS) --features $(RULESYNC_FEATURES)

rulesync: rulesync-import rulesync-generate ## rulesync: import + generate

format: ## prettier: 対象ファイルを整形(.prettierignore で除外管理)
	npx -y prettier@$(PRETTIER_VERSION) --write .

format-check: ## prettier: 整形差分をチェックのみ(CI/事前検証用)
	npx -y prettier@$(PRETTIER_VERSION) --check .

setup: ## 企画T向け初回セットアップ(対話型)
	@./scripts/setup.sh

doctor: ## 環境診断
	@./scripts/doctor.sh
