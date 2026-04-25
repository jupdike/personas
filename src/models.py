from util import last_path_component

EDGE_OF_REALISM_MODEL_PATH = "models/edgeOfRealism_eorV20BakedVAE.safetensors"
EPIC_REALISM_MODEL_PATH = "models/epicrealism_naturalSinRC1VAE.safetensors"
EPIC_PHOTOG_MODEL_PATH = "models/epicphotogasm_zUniversal.safetensors"
REALISTIC_VISION_MODEL_PATH = "models/Realistic_Vision_V5_1.safetensors"

# pick a model
#model_path = EDGE_OF_REALISM_MODEL_PATH
#model_path = EPIC_REALISM_MODEL_PATH
#model_path = REALISTIC_VISION_MODEL_PATH
model_path = EPIC_PHOTOG_MODEL_PATH
# end pick a model

model_name = last_path_component(model_path)
print("<> Using model:", model_name)
