from bootleg.annotator import Annotator
from bootleg.utils.parser_utils import get_full_config

# modified from https://guillaumegenthial.github.io/serving.html
# https://github.com/guillaumegenthial/api_ner/blob/master/serve.py

def align_data(data):
    """Given dict with lists, creates aligned strings
    Adapted from Assignment 3 of CS224N
    Args:
        data: (dict) data["x"] = ["I", "love", "you"]
              (dict) data["y"] = ["O", "O", "O"]
    Returns:
        data_aligned: (dict) data_align["x"] = "I love you"
                           data_align["y"] = "O O    O  "
    """
    spacings = [max([len(seq[i]) for seq in data.values()])
                for i in range(len(data[list(data.keys())[0]]))]
    data_aligned = dict()

    # for each entry, create aligned string
    for key, seq in data.items():
        str_aligned = ""
        for token, spacing in zip(seq, spacings):
            str_aligned += token + " " * (spacing - len(token) + 1)

        data_aligned[key] = str_aligned

    return data_aligned

def format_data(data, input_data):
    sentence = input_data.split(" ")
    probs = []
    qids = []
    entities = []
    text = []
    is_ent = []
    words = []
    ent_idx = 0
    for word_idx, word in enumerate(sentence):
        if ent_idx >= len(data["entities"]):
            # add remaining words
            probs.append("")
            qids.append("")
            entities.append("")
            text.append(" ".join(sentence[word_idx:]))
            is_ent.append(False)
            break

        start_idx = int(data["start"][ent_idx])
        end_idx = int(data["end"][ent_idx])

        if word_idx < start_idx:
            words.append(word)
            print(words)
        elif word_idx == start_idx:
            # end of phrase
            probs.append("")
            qids.append("")
            entities.append("")
            is_ent.append(False)
            text.append(" ".join(words))
            # reset words
            words = []

        if word_idx == end_idx-1:
            text.append(' '.join(sentence[start_idx:end_idx]))
            probs.append(str(round(data["probs"][ent_idx], 3)))
            qids.append(data["qids"][ent_idx])
            entities.append(data["entities"][ent_idx])
            is_ent.append(True)
            ent_idx += 1

    return {"text": text, "probs": probs, "qids": qids, "entities": entities, "is_ent": is_ent}

def get_model_api():
    """Returns lambda function for api"""
    # 1. initialize model once using annotator class
    root_dir = '/Users/meganleszczynski/bootleg'
    use_cpu = True

    cand_map = f'{root_dir}/data/wiki_entity_data/entity_mappings/alias2qids_wiki.json'
    # config_path = f'{root_dir}/models/bootleg_wiki_kg/bootleg_config.json'
    config_path = f'{root_dir}/models/bootleg_wiki/bootleg_config.json'
    config_args = get_full_config(config_path)
    # config_args.run_config.init_checkpoint = f'{root_dir}/models/bootleg_wiki_kg/bootleg_kg.pt'
    config_args.run_config.init_checkpoint = f'{root_dir}/models/bootleg_wiki/bootleg_model.pt'
    config_args.data_config.entity_dir = f'{root_dir}/data/wiki_entity_data'
    config_args.data_config.alias_cand_map = 'alias2qids_wiki.json'
    config_args.data_config.emb_dir =  f'{root_dir}/data/emb_data'
    config_args.data_config.word_embedding.cache_dir =  f'{root_dir}/pretrained_bert_models'
    config_args.run_config.cpu = use_cpu

    ann = Annotator(config_args=config_args, cand_map=cand_map, device='cuda' if not use_cpu else 'cpu')
    print("Loaded model")

    def model_api(input_data):
        output = ann.label_mentions(input_data)
        print(output)
        formatted_output = format_data(output, input_data)
        print(formatted_output)
        text_chunks = formatted_output["text"]
        is_ent = formatted_output["is_ent"]
        # del formatted_output["is_ent"]
        # aligned_data = align_data(formatted_output)
        # aligned_data["text_chunks"] = text_chunks
        # aligned_data["is_ent"] = is_ent
        # print(aligned_data)
        # return aligned_data
        return formatted_output

    return model_api