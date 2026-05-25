"""品質評価の採点ロジック。

docs/06-evaluation.md §7 に準拠する自動採点:
- Coding: ```python ブロック抽出 → exec → 関数を取り出して tests を回す
- Summary: 期待キーポイント・決定・TODO の n-gram 被覆率
- Needle: 出力に needle_id が含まれるかの文字列マッチ

LLM-as-a-judge は使わない。
"""

from __future__ import annotations

import re
import signal
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any

# ---------- 共通 ----------

CODE_FENCE_PY = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_python_code(text: str) -> str | None:
    """応答テキストから ```python ... ``` を抜き出す。複数あれば結合。"""
    matches = CODE_FENCE_PY.findall(text)
    if not matches:
        return None
    return "\n\n".join(m.strip() for m in matches)


@contextmanager
def _alarm(seconds: int):
    """SIGALRM で関数実行を時間制限する。Unix 系専用。"""

    def handler(signum: int, frame: Any) -> None:  # noqa: ARG001
        raise TimeoutError("execution timed out")

    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# ---------- Coding ----------


@dataclass
class CodingCaseResult:
    args: list[Any]
    kwargs: dict[str, Any]
    expected: Any
    got: Any | None
    passed: bool
    error: str | None = None


@dataclass
class CodingTaskResult:
    task_id: str
    task_type: str
    function_name: str
    code_extracted: bool
    exec_error: str | None
    cases: list[CodingCaseResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.cases if c.passed)

    @property
    def total(self) -> int:
        return len(self.cases)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["passed"] = self.passed
        d["total"] = self.total
        d["pass_rate"] = round(self.pass_rate, 3)
        return d


def score_coding_task(
    task: dict[str, Any], response: str, *, exec_timeout_sec: int = 5
) -> CodingTaskResult:
    """1 件のコーディングタスクを採点する。

    task の形式:
    {
        "task_id": "...",
        "task_type": "impl | bugfix | refactor | ...",
        "function_name": "fn",
        "tests": [
            {"args": [...], "kwargs": {...}?, "expected": ...}, ...
        ]
    }
    """
    code = extract_python_code(response)
    result = CodingTaskResult(
        task_id=task["task_id"],
        task_type=task.get("task_type", ""),
        function_name=task["function_name"],
        code_extracted=code is not None,
        exec_error=None,
    )
    if code is None:
        return result

    ns: dict[str, Any] = {}
    try:
        with _alarm(exec_timeout_sec):
            exec(compile(code, "<llm-output>", "exec"), ns)
    except (Exception, TimeoutError) as e:  # noqa: BLE001
        result.exec_error = f"{type(e).__name__}: {e}"
        return result

    func = ns.get(task["function_name"])
    if not callable(func):
        result.exec_error = f"function '{task['function_name']}' not found in extracted code"
        return result

    for case in task.get("tests", []):
        args = case.get("args", [])
        kwargs = case.get("kwargs", {})
        expected = case.get("expected")
        try:
            with _alarm(exec_timeout_sec):
                got = func(*args, **kwargs)
            passed = got == expected
            result.cases.append(
                CodingCaseResult(args=args, kwargs=kwargs, expected=expected, got=got, passed=passed)
            )
        except (Exception, TimeoutError) as e:  # noqa: BLE001
            result.cases.append(
                CodingCaseResult(
                    args=args,
                    kwargs=kwargs,
                    expected=expected,
                    got=None,
                    passed=False,
                    error=f"{type(e).__name__}: {e}",
                )
            )

    return result


# ---------- Summary ----------


def _normalize(s: str) -> str:
    return re.sub(r"\s+", "", s).lower()


def _ngrams(s: str, n: int) -> set[str]:
    s = _normalize(s)
    if len(s) < n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def _coverage(expected: str, response: str, *, n: int) -> float:
    """expected の n-gram のうち、response にも現れる割合(0..1)。"""
    exp = _ngrams(expected, n)
    res = _ngrams(response, n)
    if not exp:
        return 0.0
    return len(exp & res) / len(exp)


@dataclass
class SummaryItemResult:
    text: str
    coverage: float
    matched: bool


@dataclass
class SummaryCategoryResult:
    category: str
    items: list[SummaryItemResult] = field(default_factory=list)

    @property
    def matched_count(self) -> int:
        return sum(1 for i in self.items if i.matched)

    @property
    def total(self) -> int:
        return len(self.items)

    @property
    def match_rate(self) -> float:
        return self.matched_count / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "matched_count": self.matched_count,
            "total": self.total,
            "match_rate": round(self.match_rate, 3),
            "items": [asdict(i) for i in self.items],
        }


@dataclass
class SummaryTaskResult:
    task_id: str
    categories: list[SummaryCategoryResult] = field(default_factory=list)

    @property
    def overall(self) -> float:
        rates = [c.match_rate for c in self.categories if c.total > 0]
        return sum(rates) / len(rates) if rates else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "overall_match_rate": round(self.overall, 3),
            "categories": [c.to_dict() for c in self.categories],
        }


def score_summary_task(
    task: dict[str, Any],
    response: str,
    *,
    ngram: int = 3,
    threshold: float = 0.5,
) -> SummaryTaskResult:
    """会議要約タスクを採点する。

    task の形式:
    {
        "task_id": "...",
        "transcript": "...",
        "expected_keypoints": [...],
        "expected_decisions": [...],
        "expected_todos": [...],
    }

    threshold: n-gram 被覆率がこれ以上なら matched と判定。
    """
    out = SummaryTaskResult(task_id=task["task_id"])
    category_keys = {
        "keypoints": "expected_keypoints",
        "decisions": "expected_decisions",
        "todos": "expected_todos",
    }
    for cat_name, key in category_keys.items():
        items_expected = task.get(key, []) or []
        category = SummaryCategoryResult(category=cat_name)
        for item in items_expected:
            cov = _coverage(item, response, n=ngram)
            category.items.append(
                SummaryItemResult(text=item, coverage=round(cov, 3), matched=cov >= threshold)
            )
        out.categories.append(category)
    return out


# ---------- Needle ----------


def score_needle(response: str, needle_id: str) -> bool:
    return needle_id in response
