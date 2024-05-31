from safetensors.torch import load_file, save_file
clip_path = 'siglip-finetune/so400m_epoch18_aug'
model = load_file(clip_path + '/model.safetensors')
model = {k.replace('_orig_mod.', ''): v for k, v in model.items()}
save_file(model, clip_path + '/new_model.safetensors', metadata={'format': 'pt'})