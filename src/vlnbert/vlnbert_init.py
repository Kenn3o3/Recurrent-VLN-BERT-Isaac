# Modified from Recurrent VLN-BERT, 2020, Yicong.Hong@anu.edu.au

from transformers import BertConfig, BertTokenizer

from transformers import logging
def get_tokenizer(args):
    tokenizer_class = BertTokenizer
    tokenizer = tokenizer_class.from_pretrained('bert-base-uncased')
    return tokenizer

def get_vlnbert_models(config=None):
    config_class = BertConfig
    from vlnbert.vlnbert_PREVALENT import VLNBert
    model_class = VLNBert
    model_name_or_path = '/home/prj21/fyp/Recurrent-VLN-BERT-Isaac/pretrained_weight/pytorch_model.bin'
    vis_config = config_class.from_pretrained('bert-base-uncased')
    # vis_config.img_feature_dim = 2176 # Original model dim
    vis_config.img_feature_dim = 4096
    vis_config.img_feature_type = ""
    vis_config.vl_layers = 4
    vis_config.la_layers = 9
    logging.set_verbosity_error()
    visual_model = model_class.from_pretrained(model_name_or_path, config=vis_config, ignore_mismatched_sizes=True) # The mismatched visn_fc.weight -> randomly initialized values -> fine-tuned during training to adapt to the new 4096-dimensional input features.

    return visual_model
