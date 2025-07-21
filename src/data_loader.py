from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

def load_persian_qa_dataset():
    """Loads the SajjadAyoubi/persian_qa dataset from Hugging Face."""
    return load_dataset("SajjadAyoubi/persian_qa")

def _calculate_answer_length(example):
    """Helper function to calculate the length of the first answer."""
    if example["answers"]["text"]:
        return {"answer_length": len(example["answers"]["text"][0])}
    return {"answer_length": 0}

def get_filtered_datasets(dataset):
    """
    Filters the validation set into four subsets for experiments:
    1.  Questions with answers.
    2.  Questions with no answers.
    3.  Questions with long answers.
    4.  Questions with short answers.
    """
    # Add answer length column to perform filtering
    dataset_with_lengths = dataset.map(_calculate_answer_length)
    validation_lengths = dataset_with_lengths["validation"]["answer_length"]
    mean_validation_length = np.mean(validation_lengths)
    
    print(f"Mean answer length in validation set: {mean_validation_length:.2f} characters")

    has_answer_validation = dataset_with_lengths["validation"].filter(lambda x: x["answer_length"] > 0)
    no_answer_validation = dataset_with_lengths["validation"].filter(lambda x: x["answer_length"] == 0)

    long_answer_validation = has_answer_validation.filter(lambda x: x["answer_length"] > mean_validation_length)
    short_answer_validation = has_answer_validation.filter(lambda x: x["answer_length"] <= mean_validation_length)
    
    return {
        "has_answer": has_answer_validation,
        "no_answer": no_answer_validation,
        "long_answer": long_answer_validation,
        "short_answer": short_answer_validation
    }

def preprocess_data(examples, tokenizer, max_length=384, doc_stride=128):
    """Tokenizes and prepares the dataset for question answering models."""
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples["offset_mapping"]
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        if not answers["answer_start"]:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)
    return tokenized_examples