import argparse
import evaluate
from transformers import AutoTokenizer, TrainingArguments, Trainer
from src.data_loader import load_persian_qa_dataset, preprocess_data
from src.model import get_Youtubeing_model
from src.utils import postprocess_qa_predictions

def compute_metrics(p, raw_validation_set, tokenized_validation_set, tokenizer):
    raw_predictions = p.predictions
    final_predictions = postprocess_qa_predictions(
        raw_validation_set,
        tokenized_validation_set,
        raw_predictions,
        tokenizer
    )

    formatted_predictions = [
        {"id": str(k), "prediction_text": v["text"], "no_answer_probability": float(v["null_score"])}
        for k, v in final_predictions.items()
    ]
    references = [{"id": str(ex["id"]), "answers": ex["answers"]} for ex in raw_validation_set]
    metric = evaluate.load("squad_v2")
    return metric.compute(predictions=formatted_predictions, references=references)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model for Persian Question Answering.")
    parser.add_argument("--model_name", type=str, required=True, help="Hugging Face model name.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine-tuned model.")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for parameter-efficient fine-tuning.")
    args = parser.parse_args()

    print(f"Starting fine-tuning for model: {args.model_name}")
    print(f"Using LoRA: {args.use_lora}")

    dataset = load_persian_qa_dataset()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    tokenized_datasets = dataset.map(
        lambda x: preprocess_data(x, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    raw_validation_set = dataset["validation"]
    
    temp_tokenized = raw_validation_set.map(
        lambda x: preprocess_data(x, tokenizer, return_overflowing_tokens=True),
        batched=True, remove_columns=raw_validation_set.column_names
    )
    overflow_mapping = temp_tokenized.pop("overflow_to_sample_mapping")
    example_ids = [raw_validation_set[i]['id'] for i in overflow_mapping]
    tokenized_validation_set = tokenized_datasets['validation'].add_column('example_id', example_ids)

    model = get_Youtubeing_model(args.model_name, use_lora=args.use_lora)
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_validation_set,
        tokenizer=tokenizer,
        compute_metrics=lambda p: compute_metrics(p, raw_validation_set, tokenized_validation_set, tokenizer),
    )

    trainer.train()
    print(f"Training complete. Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()