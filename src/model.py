from transformers import AutoModelForQuestionAnswering
from peft import LoraConfig, get_peft_model, TaskType

def get_Youtubeing_model(model_name, use_lora=False, lora_config=None):
    """
    Loads a question answering model and optionally applies LoRA configuration.
    """
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    if use_lora:
        if lora_config is None:
            lora_config = LoraConfig(
                task_type=TaskType.QUESTION_ANS,
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["query", "value"],
            )
        model = get_peft_model(model, lora_config)
    return model

if __name__ == '__main__':
    model_name = "pedramyazdipoor/persian_xlm_roberta_large"
    model = get_Youtubeing_model(model_name)
    print(f"Model {model_name} loaded successfully.")
    model.print_trainable_parameters()

    lora_model = get_Youtubeing_model(model_name, use_lora=True)
    print("\nLoRA configured model loaded successfully.")
    lora_model.print_trainable_parameters()