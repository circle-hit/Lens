import json
import os
import random
import datasets
from hashlib import md5
import ast

logger = datasets.logging.get_logger(__name__)
TASK_CONFIG_FILES = {"train": "train_tasks.json", "dev": "dev_tasks.json", "test": "test_tasks.json"}
INSTRUCTION_STRATEGIES = ['single', 'multiple']
# ANSWER_PREFIX = "[/INST] "
ANSWER_PREFIX = "assistant \n\n"
SINGLE_QUOTES_SUBSTITUTE = "#$%#"

phi3_prompt = """<|system|>
{sys_prompt}<|end|>
<|user|>
{content}<|end|>
<|assistant|>
"""

# DEFAULT_SYSTEM_PROMPT = """"""
DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant."""


def gen_cache_path(cache_dir, data_args):
    hash_str = data_args.data_dir + data_args.task_config_dir + \
               data_args.instruction_file + data_args.instruction_strategy + \
               str(data_args.max_num_instances_per_task) + str(data_args.max_num_instances_per_eval_task)
    hash_obj = md5(hash_str.encode("utf-8"))
    hash_id = hash_obj.hexdigest()
    cache_path = os.path.join(cache_dir, str(hash_id))

    return cache_path


def check_path(path):
    if not path or not os.path.exists(path):
        raise ValueError('{} is not valid, please check the input path!'.format(path))


def save_ds(instances, file_name):
    with open(file_name, "w+", encoding='utf-8') as fi:
        json.dump(instances, fi, ensure_ascii=False, indent=2)


class DataConfig(datasets.BuilderConfig):
    """
    Config dataset load procedure.

    Args:
        data_dir: task data dir, which contains the corresponding dataset dirs
        prompt_path: prompt json file, which saves task and its prompts map
        task_file: task config file, save training and testing split config, and sampling strategies.
         Support two sampling strategies: 'random' indicates random sampling, while 'full' means to return all samples.
        max_num_instances_per_task: max training sample size of each task
        max_num_instances_per_eval_task: max eval sample size of each task
    """

    def __init__(
            self,
            *args,
            data_dir=None,
            over_sampling=None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.over_sampling = over_sampling


class DataInstructions(datasets.GeneratorBasedBuilder):
    """InstructData Dataset."""

    VERSION = datasets.Version("2.0.0")
    BUILDER_CONFIG_CLASS = DataConfig
    BUILDER_CONFIGS = [
        DataConfig(name="default", description="Default config for NaturalInstructions")
    ]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        LANG_LIST = ['en', 'zh', 'ja', 'ko', 'ar', 'sw', 'bn']
        info = datasets.DatasetInfo(
            features=datasets.Features(
                {
                    # "Task": datasets.Value("string"),
                    "Dataset": datasets.Value("string"),
                    "subset": datasets.Value("string"),
                    "Samples": [{
                        "id": datasets.Value("string"),
                        "sentence": datasets.Value("string"),
                        "label": datasets.Value("string"),
                        "ground_truth": datasets.Value("string")
                    }],
                    "Instance": {
                        "id": datasets.Value("string"),
                        "label": datasets.Value("string"),
                    }
                }
            ),
            supervised_keys=None
        )
        for lang in LANG_LIST:
            info.features["Instance"].update({f"instruction_{lang}": datasets.Value("string"),})
        return info

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        if self.config.data_dir is None:
            logger.error("Please provide right input: data_dir!")

        # split dir save datasets
        # task config to specify train,dev,test
        split_dir = self.config.data_dir

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "path": split_dir + '/train.json',
                    "subset": "train"
                }),
        ]


    def _load_dataset(self, dataset_path):
        with open(dataset_path, encoding="utf-8") as task_f:
            s = task_f.read()
            instances = json.loads(s)

        return instances


    def _get_instruction(self, task):
        assert self.config.instruction_strategy in INSTRUCTION_STRATEGIES
        if self.config.num_examples is not None and self.config.num_examples > 0:
            task_instructions = self.config.instructions['few-shot'][task]
        else:
            task_instructions = self.config.instructions['zero-shot'][task]
        if self.config.instruction_strategy == "single":
            return task_instructions[0]
        else:
            return random.choice(task_instructions)


    def _sampling_dataset(self, instances, sampling_strategy, max_num_instances):
        if sampling_strategy == 'random' and max_num_instances is not None and max_num_instances >= 0:
            instances = instances[:max_num_instances]
        if max_num_instances!=None and self.config.over_sampling and len(instances) < max_num_instances:
            origin_instances = instances.copy()
            while len(instances) < max_num_instances:
                instances.append(random.choice(origin_instances))

        return instances

    def load_dataset(self, dataset_path, subset):

        data = self._load_dataset(dataset_path)
        dataset_name = str(dataset_path) if type(dataset_path) is not str else dataset_path
        print("dataset_name: \n", dataset_name)
        print(list(data.keys()))

        sample_template = {"Dataset": dataset_name, "Samples": [], "subset": subset}
        lang_list = ['en', 'zh', 'ja', 'ko', 'ar', 'sw', 'bn']

        for idx, instance in enumerate(data['Instances']):
            example = sample_template.copy()
            # llama-2
            # instruction = llama2_prompt.format(
            #     system_prompt="You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.",
            #     user_message=instance["input"]
            # )

            # instruction = regular_prompt.format(
            #     question=instance["input"]
            # )
            example["Instance"] = {
                    "id": str(idx)
                }
            for lang in lang_list:
                instruction = phi3_prompt.format(
                    content=instance[f'input_{lang}'],
                    sys_prompt=DEFAULT_SYSTEM_PROMPT
                )
                label = instance["output"] + '<|end|>'

                example["Instance"].update({
                    "label": label,
                    f"instruction_{lang}": instruction,
                })

            yield example


    def _generate_examples(self, path=None, subset=None):
        """Yields examples."""
        logger.info(f"Generating tasks from = {path}")

        # load dataset
        idx = -1
        instances = []
        for sample in self.load_dataset(path, subset):
            idx += 1
            instances.append(sample)
            yield f"{path}##{idx}", sample
