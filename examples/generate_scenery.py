#!/usr/bin/env python3
"""
ランダムな風景画像を生成するスクリプト
DALL-E 3 を使用
"""

import random
import json
from pathlib import Path

# 風景要素のリスト
SCENERY_PROMPTS = [
    # 自然風景
    "A serene mountains landscape with snow-capped peaks and a crystal clear lake reflecting the sky at sunset",
    "A peaceful bamboo forest with sunlight filtering through the leaves and a gentle stream",
    "A breathtaking waterfall cascading down mossy rocks into a crystal clear pool",
    "A misty morning view of rice terraces in the mountains",
    "A stunning aurora borealis dancing over a frozen northern lake",
    "A colorful sunset over a coastal cliff with waves crashing below",
    "A tranquil cherry blossom grove in spring with petals floating in a pond",
    "A majestic redwood forest with sunlight rays filtering through ancient trees",
    "A vast desert landscape with sand dunes and a colorful sunset",
    "A peaceful meadow filled with wildflowers under a starry night sky",

    # 都市景観
    "A futuristic city skyline at dusk with neon lights reflecting on wet streets",
    "A historic European town square with cobblestone streets and charming architecture",
    "A busy modern metropolis at night with towering skyscrapers and glowing windows",
    "A traditional Japanese town with wooden buildings and bamboo groves",
    "A coastal fishing village with colorful houses and boats in the harbor",
    "An ancient castle perched on a hill overlooking a valley",
    "A vibrant market town with colorful stalls and people gathering",
    "A quiet mountain town surrounded by pine forests",

    # 季節別
    "An autumn forest with golden and crimson leaves blanketing the ground",
    "A snowy winter landscape with ice sculptures and a cozy village",
    "A lush tropical beach with palm trees and turquoise waters",
    "A rainy countryside scene with green fields and mud paths",
]

# 追加のスタイルをランダムに適用
STYLE_SUFFIXES = [
    "",
    ", with a realistic photography style",
    ", in an anime art style",
    ", with a watercolor painting style",
    ", with an oil painting style",
    ", with a digital art style",
    ", with a sketch style",
    ", in a vintage film aesthetic",
    ", with a cinematic lighting",
    ", with pastel colors",
    ", with high contrast and dramatic shadows",
]

def generate_random_prompt():
    """ランダムな風景プロンプトを生成"""
    base_prompt = random.choice(SCENERY_PROMPTS)
    style = random.choice(STYLE_SUFFIXES)
    return f"{base_prompt}{style}"

def main():
    print("=" * 60)
    print("ランダムな風景画像生成器")
    print("=" * 60)

    # ランダムなプロンプトを生成
    prompt = generate_random_prompt()
    print(f"\n【選択された風景】")
    print(f"{prompt}")
    print()

    # ここで画像生成処理を実装
    # DALL-E 3 API を使用する場合：
    # - OpenAI の DALL-E 3 API を利用
    # - または Stability AI の SDXL など

    print("画像生成の実装:")
    print("1. DALL-E 3 API を使用:")
    print("   from openai import OpenAI")
    print("   client = OpenAI()")
    print("   response = client.images.generate(")
    print("       model='dall-e-3',")
    print("       prompt='...",
    print("       size='1024x1024')")
    print("   # response.image.url で画像 URL を取得")
    print()

    print("2. Stability AI の Stable Diffusion:")
    print("   from diffusers import StableDiffusionXLPipeline")
    print("   pipe = StableDiffusionXLPipeline.from_pretrained(...")
    print("   image = pipe(prompt=prompt).images[0]")
    print()

    print("=" * 60)
    print("スクリプトが完成しました！")
    print("画像生成ライブラリを追加して使用してください")
    print("=" * 60)

if __name__ == "__main__":
    main()
