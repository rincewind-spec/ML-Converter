import coremltools as ct
import torch
from transformers import PegasusModel, PegasusConfig, PegasusTokenizer

configuration = PegasusConfig()
tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-large")
model = PegasusModel.from_pretrained("google/pegasus-large", torchscript=True)
model.eval()
example_input = "PG&E stated it scheduled the blackouts in response to forecasts for high winds amid dry conditions. " \
                "The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were scheduled to be " \
                "affected by the shutoffs which were expected to last through at least midday tomorrow."
traced_model = torch.jit.trace(model, example_input)
ml_model = ct.convert(traced_model, inputs=[tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='tf')])
ml_model.save("TextSummarizer.mlmodel")
