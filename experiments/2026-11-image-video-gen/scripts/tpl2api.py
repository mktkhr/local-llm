#!/usr/bin/env python3
"""ComfyUI の UI 形式ワークフロー(公式テンプレート)を API 形式(/prompt に渡す dict)に変換する。

UI 形式はノードの widget 値を順序付き配列で持ち、API 形式は入力名をキーにした dict で持つ。
対応付けには /object_info(各ノード型の入力定義の順序)が要る。サブグラフは平坦化し、
ノード id は "<外側id>:<内側id>" にする。

  python3 tpl2api.py --template image_z_image_turbo.json --object-info object_info.json \
      --prompt "..." --seed 42 --width 1024 --height 1024 --set KSampler.steps=8 > graph.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

SKIP_TYPES = {"MarkdownNote", "Note", "Reroute", "PrimitiveNode"}
PASSTHROUGH_TYPES = {"PreviewAny"}  # 入力をそのまま出力に流すだけの表示ノード
CONTROL_VALUES = {"fixed", "increment", "decrement", "randomize"}
WIDGET_TYPES = {"STRING", "INT", "FLOAT", "BOOLEAN"}
SUBGRAPH_IN, SUBGRAPH_OUT = -10, -20


def _is_widget_spec(spec) -> bool:
    """object_info の入力定義が widget(値を直接持つ)か、接続(MODEL/CLIP など)か。"""
    if not isinstance(spec, (list, tuple)) or not spec:
        return False
    t = spec[0]
    if isinstance(t, list):  # COMBO
        return True
    if isinstance(t, str):
        if t in WIDGET_TYPES:
            return True
        if t == "COMBO":
            return True
    return False


def _input_specs(object_info: dict, node_type: str) -> list[tuple[str, object]]:
    d = (object_info.get(node_type) or {}).get("input") or {}
    specs = list((d.get("required") or {}).items()) + list((d.get("optional") or {}).items())
    return specs


def flatten(workflow: dict) -> tuple[dict[str, dict], list[tuple[str, int, str, int]], dict[tuple[str, int], object]]:
    """サブグラフを展開し、{node_key: node}、接続 [(from_key, from_slot, to_key, to_slot)]、
    リテラル {(node_key, slot): value}(サブグラフノード側に昇格した widget 値で、外から未接続の入力)を返す。

    サブグラフノードの入出力はスロット番号ではなく名前で対応付ける(ノード側の inputs の並びは
    定義側の inputs の並びと一致しないことがある)。
    """
    subgraphs = {sg["id"]: sg for sg in ((workflow.get("definitions") or {}).get("subgraphs") or [])}
    nodes: dict[str, dict] = {}
    links: list[tuple[str, int, str, int]] = []
    literals: dict[tuple[str, int], object] = {}

    def promoted(sn: dict, sg: dict) -> dict[str, object]:
        """サブグラフノードの widgets_values は、サブグラフ定義の inputs のうち widget 型(STRING/INT/FLOAT/BOOLEAN/COMBO)
        のものの並び順。seed の直後に control_after_generate が挟まることがある。"""
        wv = sn.get("widgets_values") if isinstance(sn.get("widgets_values"), list) else []
        names = [i["name"] for i in sg.get("inputs") or []
                 if isinstance(i.get("type"), str) and (i["type"] in WIDGET_TYPES or i["type"].startswith("COMBO"))]
        out: dict[str, object] = {}
        pos = 0
        for name in names:
            if pos >= len(wv):
                break
            out[name] = wv[pos]
            pos += 1
            if name in ("seed", "noise_seed") and pos < len(wv) and wv[pos] in CONTROL_VALUES:
                pos += 1
        return out

    def expand(node_list: list[dict], link_list: list, prefix: str, outer_in: dict[str, tuple[str, int]],
               outer_out: dict[str, list[tuple[str, int]]], outer_lit: dict[str, object], sg_def: dict | None):
        norm = []
        for ln in link_list:
            if isinstance(ln, dict):
                norm.append((ln["origin_id"], ln["origin_slot"], ln["target_id"], ln["target_slot"]))
            else:
                norm.append((ln[1], ln[2], ln[3], ln[4]))
        local_sub: dict[int, dict] = {}
        for n in node_list:
            if n["type"] in subgraphs:
                if n.get("mode") in (2, 4):
                    continue  # ミュート/バイパスされたサブグラフは丸ごと落とす
                local_sub[n["id"]] = n
            else:
                nodes[f"{prefix}{n['id']}"] = n

        def in_name(node: dict, slot: int) -> str:
            return node["inputs"][slot]["name"]

        def out_name(node: dict, slot: int) -> str:
            return node["outputs"][slot]["name"]

        # サブグラフノードへの入力 {(sub_id, input_name): src} / サブグラフノードからの出力 {(sub_id, output_name): [tgt]}
        sub_in: dict[tuple[int, str], tuple[str, int]] = {}
        sub_out: dict[tuple[int, str], list[tuple[str, int]]] = {}
        sub_lit: dict[tuple[int, str], object] = {}
        for (fo, fs, to, ts) in norm:
            if fo == SUBGRAPH_IN:
                name = sg_def["inputs"][fs]["name"] if sg_def else None
                src = outer_in.get(name)
                if src is None:
                    if name in outer_lit:  # 外側ノードに昇格した widget 値
                        if to in local_sub:
                            sub_lit[(to, in_name(local_sub[to], ts))] = outer_lit[name]
                        else:
                            literals[(f"{prefix}{to}", ts)] = outer_lit[name]
                    continue  # 未接続 → 内側ノード自身の widget 値を使う
                if to in local_sub:
                    sub_in[(to, in_name(local_sub[to], ts))] = src
                else:
                    links.append((src[0], src[1], f"{prefix}{to}", ts))
            elif to == SUBGRAPH_OUT:
                for (tk, tslot) in (outer_out.get(sg_def["outputs"][ts]["name"], []) if sg_def else []):
                    if fo in local_sub:
                        sub_out.setdefault((fo, out_name(local_sub[fo], fs)), []).append((tk, tslot))
                    else:
                        links.append((f"{prefix}{fo}", fs, tk, tslot))
            elif fo in local_sub and to in local_sub:
                raise NotImplementedError("subgraph→subgraph の直結は未対応")
            elif to in local_sub:
                sub_in[(to, in_name(local_sub[to], ts))] = (f"{prefix}{fo}", fs)
            elif fo in local_sub:
                sub_out.setdefault((fo, out_name(local_sub[fo], fs)), []).append((f"{prefix}{to}", ts))
            elif fo in (n["id"] for n in node_list) and to in (n["id"] for n in node_list):
                links.append((f"{prefix}{fo}", fs, f"{prefix}{to}", ts))
        for sid, sn in local_sub.items():
            sg = subgraphs[sn["type"]]
            oin = {name: src for (nid, name), src in sub_in.items() if nid == sid}
            oout: dict[str, list[tuple[str, int]]] = {}
            for (nid, name), tgts in sub_out.items():
                if nid == sid:
                    oout.setdefault(name, []).extend(tgts)
            olit = promoted(sn, sg)
            olit.update({name: v for (nid, name), v in sub_lit.items() if nid == sid})
            expand(sg["nodes"], sg.get("links") or [], f"{prefix}{sid}:", oin, oout, olit, sg)

    expand(workflow["nodes"], workflow.get("links") or [], "", {}, {}, {}, None)
    return nodes, links, literals


def _widget_inputs(specs: list[tuple[str, object]], wv: list, prefix: str = "") -> dict:
    """widgets_values を object_info の入力順に割り当てる。seed の直後の control_after_generate は読み飛ばす。
    COMFY_DYNAMICCOMBO_V3 は選択キーの後に、そのオプションの入力が続く(API では "name.sub" で平坦化)。"""
    out: dict = {}
    pos = 0

    def take(specs, prefix):
        nonlocal pos
        for name, spec in specs:
            if pos >= len(wv):
                return
            if not _is_widget_spec(spec) and not (isinstance(spec[0], str) and spec[0] == "COMFY_DYNAMICCOMBO_V3"):
                continue
            key = f"{prefix}{name}"
            out[key] = wv[pos]
            pos += 1
            if name in ("seed", "noise_seed") and pos < len(wv) and wv[pos] in CONTROL_VALUES:
                pos += 1
            if isinstance(spec[0], str) and spec[0] == "COMFY_DYNAMICCOMBO_V3":
                opt = next((o for o in spec[1].get("options", []) if o.get("key") == out[key]), None)
                if opt:
                    sub = opt.get("inputs") or {}
                    take(list((sub.get("required") or {}).items()) + list((sub.get("optional") or {}).items()), key + ".")

    take(specs, prefix)
    return out


def to_api(workflow: dict, object_info: dict, prune: bool = True, sets: list[str] | None = None,
           enable: list[int] | None = None) -> dict:
    """UI 形式 → API 形式。バイパス(mode 4)ノードは同型の入力へ素通し、ミュート(mode 2)は切断する。
    prune=True なら定数スイッチ(ComfySwitchNode)を畳み込み、出力ノードから辿れないノードを捨てる。
    sets は畳み込みの前に適用する(例: "154:value=false" でスイッチを切替)。
    enable はトップレベルのノード id の一覧で、そのバイパス/ミュートを解除してから変換する(公式テンプレートが
    バイパスで同梱している超解像などの枝を有効にする)。"""
    if enable:
        workflow = json.loads(json.dumps(workflow))
        for n in workflow["nodes"]:
            if n["id"] in set(enable):
                n["mode"] = 0
    nodes, links, literals = flatten(workflow)
    incoming: dict[tuple[str, int], tuple[str, int]] = {(tk, ts): (fk, fs) for (fk, fs, tk, ts) in links}

    def resolve(key: str, slot: int, depth: int = 0):
        """入力 (key, slot) の実際の供給元。("link", (node_key, out_slot)) / ("lit", value) / None。
        バイパスノードと PreviewAny(表示専用、素通し)は飛ばす。"""
        if depth > 64:
            return None
        src = incoming.get((key, slot))
        if src is None:
            return ("lit", literals[(key, slot)]) if (key, slot) in literals else None
        n = nodes.get(src[0])
        if n is None or n.get("mode") == 2:
            return None
        if n["type"] in PASSTHROUGH_TYPES:
            return resolve(src[0], 0, depth + 1)
        if n["type"] in SKIP_TYPES:
            return None
        if n.get("mode") == 4:
            typ = (n.get("outputs") or [{}])[src[1]].get("type") if src[1] < len(n.get("outputs") or []) else None
            for islot, inp in enumerate(n.get("inputs") or []):
                if inp.get("type") == typ:
                    return resolve(src[0], islot, depth + 1)
            return None
        return ("link", src)

    api: dict[str, dict] = {}
    for key, n in nodes.items():
        t = n["type"]
        if t in SKIP_TYPES or t in PASSTHROUGH_TYPES or n.get("mode") in (2, 4):
            continue
        if t not in object_info:
            raise KeyError(f"object_info に無いノード型: {t}")
        wv = n.get("widgets_values") if isinstance(n.get("widgets_values"), list) else []
        inputs = _widget_inputs(_input_specs(object_info, t), wv)
        for slot, inp in enumerate(n.get("inputs") or []):
            r = resolve(key, slot)
            if r is None:
                continue
            inputs[inp["name"]] = [r[1][0], r[1][1]] if r[0] == "link" else r[1]
        api[key] = {"class_type": t, "inputs": inputs}
    if sets:
        apply_overrides(api, sets=sets)
    if prune:
        fold_switches(api)
        prune_unreachable(api, object_info)
    return api


def fold_switches(api: dict) -> None:
    """switch が定数(リテラル、または PrimitiveBoolean の value)の ComfySwitchNode を取り除き、
    その出力の参照を選ばれた側の供給元に付け替える。"""
    def const_bool(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, list) and v[0] in api and api[v[0]]["class_type"] == "PrimitiveBoolean":
            return bool(api[v[0]]["inputs"].get("value"))
        return None

    changed = True
    while changed:
        changed = False
        for k, v in list(api.items()):
            if v["class_type"] != "ComfySwitchNode":
                continue
            b = const_bool(v["inputs"].get("switch"))
            if b is None:
                continue
            src = v["inputs"].get("on_true" if b else "on_false")
            for ov in api.values():
                for name, x in list(ov["inputs"].items()):
                    if isinstance(x, list) and x[0] == k:
                        if src is None:
                            del ov["inputs"][name]
                        else:  # 接続 [key, slot] ならコピー、リテラルならそのまま
                            ov["inputs"][name] = list(src) if isinstance(src, list) else src
            del api[k]
            changed = True


def prune_unreachable(api: dict, object_info: dict) -> None:
    roots = [k for k, v in api.items() if (object_info.get(v["class_type"]) or {}).get("output_node")]
    keep: set[str] = set()
    stack = list(roots)
    while stack:
        k = stack.pop()
        if k in keep:
            continue
        keep.add(k)
        for x in api[k]["inputs"].values():
            if isinstance(x, list) and x[0] in api:
                stack.append(x[0])
    for k in list(api):
        if k not in keep:
            del api[k]


# -------------------------------------------------------------------- 上書き
TEXT_INPUTS = ("text", "prompt", "positive", "value", "string")
FOLLOW_INPUTS = ("positive", "conditioning", "source", "text", "prompt", "string", "value", "on_false", "on_true", "replace")
SIZE_TYPES_HINT = ("Latent", "Resolution", "Empty")


def find_prompt_input(api: dict) -> tuple[str, str]:
    """サンプラーの positive から上流へ辿り、最初に出てくる文字列リテラルの (node_key, input_name) を返す。"""
    starts = [k for k, v in api.items() if isinstance(v["inputs"].get("positive"), list)]
    seen: set[str] = set()
    stack = [api[k]["inputs"]["positive"][0] for k in starts]
    while stack:
        k = stack.pop(0)
        if k in seen or k not in api:
            continue
        seen.add(k)
        ins = api[k]["inputs"]
        for name in TEXT_INPUTS:
            if isinstance(ins.get(name), str):
                return k, name
        for name in FOLLOW_INPUTS:
            if isinstance(ins.get(name), list):
                stack.append(ins[name][0])
    raise ValueError("prompt を入れる先が見つかりません")


def apply_overrides(api: dict, prompt: str | None = None, seed: int | None = None, width: int | None = None,
                    height: int | None = None, steps: int | None = None, sets: list[str] | None = None) -> dict:
    """よく変える値をノード型を知らずに差し替える。prompt は最初のテキストエンコード系ノードに入れる。"""
    def nodes_with(pred):
        return [(k, v) for k, v in api.items() if pred(v)]

    if prompt is not None:
        k, name = find_prompt_input(api)
        api[k]["inputs"][name] = prompt
    if seed is not None:
        for _, v in nodes_with(lambda v: any(i in v["inputs"] and not isinstance(v["inputs"][i], list) for i in ("seed", "noise_seed"))):
            for i in ("seed", "noise_seed"):
                if i in v["inputs"] and not isinstance(v["inputs"][i], list):
                    v["inputs"][i] = seed
    for dim, val in (("width", width), ("height", height)):
        if val is not None:
            for _, v in nodes_with(lambda v: dim in v["inputs"] and not isinstance(v["inputs"][dim], list)
                                   and any(h in v["class_type"] for h in SIZE_TYPES_HINT)):
                v["inputs"][dim] = val
    if steps is not None:
        for _, v in nodes_with(lambda v: "steps" in v["inputs"] and not isinstance(v["inputs"]["steps"], list)):
            v["inputs"]["steps"] = steps
    for s in sets or []:
        target, _, raw = s.partition("=")
        if "." in target:
            typ, _, name = target.rpartition(".")
        else:
            typ, _, name = target.rpartition(":")
        try:  # 数値・真偽値・JSON はそのまま、それ以外は文字列
            val = json.loads(raw)
        except json.JSONDecodeError:
            val = raw
        hit = 0
        for k, v in api.items():
            if (k == typ or v["class_type"] == typ) and name in v["inputs"]:
                v["inputs"][name] = val
                hit += 1
        if not hit:
            raise ValueError(f"--set {s}: 該当するノードがありません")
    return api


def swap_unet_loader_gguf(api: dict, gguf_name: str) -> int:
    """UNETLoader を ComfyUI-GGUF の UnetLoaderGGUF に置き換える(同じキーなので下流の参照は不変)。置換数を返す。"""
    n = 0
    for v in api.values():
        if v["class_type"] == "UNETLoader":
            v["class_type"] = "UnetLoaderGGUF"
            v["inputs"] = {"unet_name": gguf_name}
            n += 1
    return n


def preview_to_save(api: dict, filename_prefix: str) -> int:
    """PreviewImage(一時ファイル)を SaveImage に置き換えて出力を残す。公式配布のワークフローに多い。置換数を返す。"""
    n = 0
    for v in api.values():
        if v["class_type"] == "PreviewImage":
            v["class_type"] = "SaveImage"
            v["inputs"] = {"images": v["inputs"].get("images"), "filename_prefix": filename_prefix}
            n += 1
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", type=pathlib.Path, required=True)
    ap.add_argument("--object-info", type=pathlib.Path, required=True)
    ap.add_argument("--prompt")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--set", action="append", default=[], help="NodeType.input=value または id:input=value(複数可)")
    ap.add_argument("--gguf", help="UNETLoader を UnetLoaderGGUF(このファイル)に置き換える")
    args = ap.parse_args(argv)
    api = to_api(json.loads(args.template.read_text()), json.loads(args.object_info.read_text()), sets=args.set)
    api = apply_overrides(api, args.prompt, args.seed, args.width, args.height, args.steps)
    if args.gguf:
        swap_unet_loader_gguf(api, args.gguf)
    json.dump(api, sys.stdout, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
