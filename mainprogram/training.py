# -*- coding: utf-8 -*-
"""
train_kogpt2.py
관광지 소개글 생성 모델 학습 스크립트 (KoGPT2 Full Fine-tuning)

- Base model : skt/kogpt2-base-v2 (약 125M, 단일 GPU 풀 파인튜닝 가능)
- 학습 데이터: 지역 / 관광지 / 소개글  (new_Token.py·corpus.py로 정제·증강한 CSV)
- 학습 방식  : Causal LM (다음 토큰 예측) 풀 파인튜닝 + CUDA(fp16)
- 결과       : ./kogpt2_finetuned 폴더에 저장 (safetensors)
              → 이후 finetuned.py / finetest.py 로 소개글 생성

실행:
    pip install torch transformers datasets accelerate
    python train_kogpt2.py --csv tourist_descriptions_cleaned.csv --epochs 5
"""

import argparse
import torch
from datasets import Dataset
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = "skt/kogpt2-base-v2"

# 추론(finetuned.py)에서 쓰는 것과 동일한 프롬프트 형식 → 학습/생성 일관성 유지
PROMPT_TMPL = "{region}의 {place}에 대한 간단한 소개글을 자연스럽게 작성하세요:"


def build_text(row, eos):
    """한 행(지역·관광지·소개글)을 '프롬프트 + 정답 소개글 + EOS' 한 문장으로."""
    prompt = PROMPT_TMPL.format(region=row["지역"], place=row["관광지"])
    return f"{prompt} {row['소개글']}{eos}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True,
                    help="지역/관광지/소개글 컬럼을 가진 학습용 CSV 경로")
    ap.add_argument("--out", default="./kogpt2_finetuned")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--push_to", default=None,
                    help="예: Junping0645/kogpt2-finetuned (HF 업로드 시)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"디바이스: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1) 토크나이저 & 모델 -----------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        bos_token="</s>", eos_token="</s>", unk_token="<unk>", pad_token="<pad>",
    )
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)

    # 2) 데이터 로드 & 텍스트 구성 --------------------------------------------
    df = pd.read_csv(args.csv)
    need = {"지역", "관광지", "소개글"}
    if not need.issubset(df.columns):
        raise ValueError(f"CSV에 {need} 컬럼이 있어야 합니다. 현재: {df.columns.tolist()}")
    df = df.dropna(subset=["지역", "관광지", "소개글"])
    eos = tokenizer.eos_token
    df["text"] = df.apply(lambda r: build_text(r, eos), axis=1)
    print(f"학습 샘플 수: {len(df)}")

    ds = Dataset.from_pandas(df[["text"]], preserve_index=False)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_len,
            padding="max_length",
        )

    ds = ds.map(tokenize, batched=True, remove_columns=["text"])
    ds = ds.train_test_split(test_size=0.1, seed=42)  # 학습/검증 분리

    # 3) Causal LM 콜레이터 (mlm=False → 다음 토큰 예측) ----------------------
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 4) 학습 설정 ------------------------------------------------------------
    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        fp16=(device == "cuda"),      # CUDA면 half precision으로 가속
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=collator,
    )

    # 5) 학습 & 저장 ----------------------------------------------------------
    trainer.train()
    trainer.save_model(args.out)          # safetensors로 저장
    tokenizer.save_pretrained(args.out)
    print(f"✅ 저장 완료: {args.out}")

    if args.push_to:
        model.push_to_hub(args.push_to)
        tokenizer.push_to_hub(args.push_to)
        print(f"✅ HF 업로드 완료: {args.push_to}")

    # 6) 간단한 생성 확인 (finetuned.py와 동일 파라미터) ----------------------
    model.eval()
    prompt = PROMPT_TMPL.format(region="안동시", place="안동 하회마을")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=200, do_sample=True,
            top_k=50, top_p=0.9, temperature=0.6,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    print("\n[생성 예시]\n", tokenizer.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
