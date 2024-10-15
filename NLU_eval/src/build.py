


def build_prmompts(dataset, dataset_name, language_name):

    if dataset_name == "xcopa":
        from template import xcopa_template as get_template
    elif dataset_name == "m-mmlu":
        from template import mmmlu_template as get_template
    elif dataset_name == "xstory":
        from template import xstory_template as get_template
    elif dataset_name == "xwino":
        from template import xwino_template as get_template
    elif dataset_name == "kmmlu":
        from template import kmmlu_template as get_template
    else:
        raise Exception("unknown dataset")

    built_prompts = []
    for data in dataset:
        built_prompts.append(get_template(data, language_name))

    return built_prompts

if __name__ == "__main__":
    data = {
        "premise": "hhh",
        "question": "sdasc",
        "choice1": "Scasc",
        "choice2": "sacxa"
    }
    print(xcopa_template(data, "zh"))
