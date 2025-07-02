import random
import vampnet_onnx as vampnet
import audiotools as at

# load the default vampnet model
interface = vampnet.interface.Interface.default()

# list available finetuned models
finetuned_model_choices = interface.available_models()
print(f"available finetuned models: {finetuned_model_choices}")

# pick a random finetuned model
model_choice = random.choice(finetuned_model_choices)
print(f"choosing model: {model_choice}")

# or pick a specific finetuned model
print(f"actually, forcing model: default")
model_choice = "default"

# load a finetuned model
interface.load_finetuned(model_choice)

# load an example audio file
signal = at.AudioSignal("assets/example.wav")
# [1, 2, 661500]

# get the tokens for the audio
codes = interface.encode(signal)
# [1, 14, 862]

# build a mask for the audio
mask = interface.build_mask(
    codes,
    signal,
    periodic_prompt=13,
    upper_codebook_mask=3,
)
# [1, 14, 862]

# generate the output tokens
output_tokens = interface.vamp(
    codes, mask, return_mask=False, temperature=1.0, typical_filtering=False, debug=True
)
# [1, 14, 862]

# convert them to a signal
# output_signal = interface.decode(codes)
output_signal = interface.decode(output_tokens)
# [1, 1, 662016]

# save the output signal
output_signal.write("scratch/output_onnx_codec.wav")
