from datasets import load_dataset, get_dataset_config_names
import random


def load_mydataset(dataset, language, num):
    dataset_dict = {
        "xcopa": "xcopa",
        "m-mmlu": "alexandrainst/m_mmlu",
        "xstory": "juletxara/xstory_cloze",
        "xwino": "Muennighoff/xwinograd",
        "kmmlu": "HAERAE-HUB/KMMLU"
    }
    split_dict = {
        "xcopa": "test",
        "m-mmlu": "test",
        "xstory": "eval",
        "xwino": "test",
        "kmmlu": "test"
    }
    
    if dataset == "kmmlu":
        # config_list = get_dataset_config_names("HAERAE-HUB/KMMLU")
        config_list = ['Accounting', 'Agricultural-Sciences', 'Aviation-Engineering-and-Maintenance', 'Biology', 'Chemical-Engineering', 'Chemistry', 'Civil-Engineering', 'Computer-Science', 'Construction', 'Criminal-Law', 'Ecology', 'Economics', 'Education', 'Electrical-Engineering', 'Electronics-Engineering', 'Energy-Management', 'Environmental-Science', 'Fashion', 'Food-Processing', 'Gas-Technology-and-Engineering', 'Geomatics', 'Health', 'Industrial-Engineer', 'Information-Technology', 'Interior-Architecture-and-Design', 'Law', 'Machine-Design-and-Manufacturing', 'Management', 'Maritime-Engineering', 'Marketing', 'Materials-Engineering', 'Mechanical-Engineering', 'Nondestructive-Testing', 'Patent', 'Political-Science-and-Sociology', 'Psychology', 'Public-Safety', 'Railway-and-Automotive-Engineering', 'Real-Estate', 'Refrigerating-Machinery', 'Social-Welfare', 'Taxation', 'Telecommunications-and-Wireless-Technology', 'Korean-History', 'Math']
        dataset = []
        for config in config_list:
            dataset.extend(load_dataset("HAERAE-HUB/KMMLU", config, download_mode='reuse_dataset_if_exists')["test"])
    else:
        dataset = list(load_dataset(dataset_dict[dataset], language, download_mode='reuse_dataset_if_exists')[split_dict[dataset]])

    try:
        random.seed(112)
        dataset = random.sample(dataset, num)
    except:
        dataset = dataset

    return dataset


if __name__ == "__main__":
    pass