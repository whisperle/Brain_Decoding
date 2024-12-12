import torch

def load_checkpoint(r_path, encoder, freeze_weights=True):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
    except Exception as e:
        print(f'Encountered exception when loading checkpoint {e}')

    try:
        pretrained_dict = checkpoint['encoder']
        msg = encoder.load_state_dict(pretrained_dict)
        print(f'loaded pretrained encoder with msg: {msg}')

        if freeze_weights:
            for param in encoder.parameters():
                param.requires_grad = False
            print("Encoder weights are frozen and will not be updated during training.")

    except Exception as e:
        print(f'Encountered exception when loading checkpoint {e}')