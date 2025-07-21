import argparse
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments
from src.data_loader import load_persian_qa_dataset, preprocess_data, get_filtered_datasets
from src.utils import postprocess_qa_predictions
import evaluate


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
    parser = argparse.ArgumentParser(description="Run experiments on a fine-tuned Persian QA model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model directory.")
    args = parser.parse_args()

    print(f"Loading model from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_path)

    dataset = load_persian_qa_dataset()
    filtered_sets = get_filtered_datasets(dataset)

    eval_args = TrainingArguments(
        output_dir=f"./{args.model_path}/eval_results",
        per_device_eval_batch_size=8,
        report_to="none",
    )

    for name, subset in filtered_sets.items():
        print(f"\n--- Running evaluation for: {name} ---")

        tokenized_subset = subset.map(
            lambda x: preprocess_data(x, tokenizer),
            batched=True,
            remove_columns=subset.column_names
        )

        temp_tokenized = subset.map(
            lambda x: preprocess_data(x, tokenizer, return_overflowing_tokens=True),
            batched=True, remove_columns=subset.column_names
        )
        overflow_mapping = temp_tokenized.pop("overflow_to_sample_mapping")
        example_ids = [subset[i]['id'] for i in overflow_mapping]
        final_eval_dataset = tokenized_subset.add_column('example_id', example_ids)

        trainer = Trainer(
            model=model,
            args=eval_args,
            eval_dataset=final_eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=lambda p: compute_metrics(p, subset, final_eval_dataset, tokenizer),
        )

        results = trainer.evaluate()
        print(f"Results for {name}: {results}")


if __name__ == "__main__":
    main()