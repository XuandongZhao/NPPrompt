from tqdm import tqdm
from openprompt.data_utils.text_classification_dataset import AgnewsProcessor, DBpediaProcessor, ImdbProcessor, AmazonProcessor, MicrosoftProcessor
from openprompt.data_utils.glue_dataset import SST2Processor, MNLIProcessor, QNLIProcessor, COLAProcessor, MRPCProcessor, QQPProcessor, RTEProcessor, STSBProcessor
from openprompt.data_utils.huggingface_dataset import YahooAnswersTopicsProcessor
import torch
import argparse
from openprompt import PromptDataLoader
from openprompt.prompts import ManualVerbalizer, EmbVerbalizer
from openprompt.prompts import ManualTemplate
from openprompt import PromptForClassification

from openprompt.utils.reproduciblity import set_seed
from openprompt.plms import load_plm
from sklearn.metrics import classification_report, matthews_corrcoef
from scipy.stats import pearsonr

parser = argparse.ArgumentParser("")
parser.add_argument("--shot", type=int, default=0)
parser.add_argument("--seed", type=int, default=144)
parser.add_argument("--plm_eval_mode", default=True, action="store_true")
parser.add_argument("--model", type=str, default='roberta')
parser.add_argument("--model_name_or_path", default='roberta-large')
parser.add_argument("--result_file", type=str, default="agnews_result.txt")
parser.add_argument("--openprompt_path", type=str, default="./")
parser.add_argument("--verbalizer", type=str, default='ept')
parser.add_argument("--calibration", default=False, action="store_true")
parser.add_argument("--nocut", default=False, action="store_true")
parser.add_argument("--filter", default="tfidf_filter", type=str)
parser.add_argument("--template_id", default=0, type=int)
parser.add_argument("--max_token_split", default=-1, type=int)
parser.add_argument("--dataset", default="agnews", type=str)
parser.add_argument("--select", default=12, type=int)
parser.add_argument("--truncate", default=-1., type=float)
parser.add_argument("--sumprob", default=False, action="store_true")
parser.add_argument("--verbose", default=True, type=bool)
parser.add_argument("--write_filter_record", default=True, action="store_true")
args = parser.parse_args()

set_seed(args.seed)
use_cuda = True
plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)

dataset = {}
if args.dataset == "agnews":
    dataset['train'] = AgnewsProcessor().get_train_examples(f"{args.openprompt_path}/datasets/TextClassification/agnews/")
    dataset['test'] = AgnewsProcessor().get_test_examples(f"{args.openprompt_path}/datasets/TextClassification/agnews/")
    class_labels = AgnewsProcessor().get_labels()
    scriptsbase = "TextClassification/agnews"
    scriptformat = "txt"
    cutoff = 0.5 if (not args.nocut) else 0.0
    max_seq_l = 128
    batch_s = 800
    num_labels = [i for i in range(4)]
elif args.dataset == "dbpedia":
    dataset['train'] = DBpediaProcessor().get_train_examples(f"{args.openprompt_path}/datasets/TextClassification/dbpedia/")
    dataset['test'] = DBpediaProcessor().get_test_examples(f"{args.openprompt_path}/datasets/TextClassification/dbpedia/")
    class_labels = DBpediaProcessor().get_labels()
    scriptsbase = "TextClassification/dbpedia"
    scriptformat = "txt"
    cutoff = 0.5 if (not args.nocut) else 0.0
    max_seq_l = 128
    batch_s = 800
    num_labels = [i for i in range(14)]
elif args.dataset == "imdb":
    # dataset['train'] = ImdbProcessor().get_train_examples(f"{args.openprompt_path}/datasets/TextClassification/imdb/")
    dataset['test'] = ImdbProcessor().get_test_examples(f"{args.openprompt_path}/datasets/TextClassification/imdb/")
    class_labels = ImdbProcessor().get_labels()
    scriptsbase = "TextClassification/imdb"
    scriptformat = "txt"
    cutoff = 0
    max_seq_l = 512
    batch_s = 60
    num_labels = [i for i in range(2)]
elif args.dataset == "amazon":
    # dataset['train'] = AmazonProcessor().get_train_examples(f"{args.openprompt_path}/datasets/TextClassification/amazon/")
    dataset['test'] = AmazonProcessor().get_test_examples(f"{args.openprompt_path}/datasets/TextClassification/amazon/")
    class_labels = AmazonProcessor().get_labels()
    scriptsbase = "TextClassification/amazon"
    scriptformat = "txt"
    cutoff = 0
    max_seq_l = 512
    batch_s = 200
    num_labels = [i for i in range(2)]
elif args.dataset == "sst2":
    # dataset['train'] = SST2Processor().get_examples('train')
    dataset['test'] = SST2Processor().get_examples('validation')
    class_labels = SST2Processor().get_labels()
    scriptsbase = "GLUE/sst2"
    scriptformat = "txt"
    cutoff = 0.5 if (not args.nocut) else 0.0
    max_seq_l = 128
    batch_s = 300
    num_labels = [i for i in range(2)]
elif args.dataset == "mnli-m":
    # dataset['train'] = MNLIProcessor('mnli-m').get_examples('train')
    dataset['test'] = MNLIProcessor('mnli_matched').get_examples('validation')
    class_labels = MNLIProcessor('mnli_matched').get_labels()
    scriptsbase = "GLUE/mnli-m"
    scriptformat = "txt"
    cutoff = 0.5 if (not args.nocut) else 0.0
    max_seq_l = 256
    batch_s = 500
    num_labels = [i for i in range(3)]
elif args.dataset == "mnli-mm":
    # dataset['train'] = MNLIProcessor('mnli-m').get_examples('train')
    dataset['test'] = MNLIProcessor('mnli_mismatched').get_examples('validation')
    class_labels = MNLIProcessor('mnli_mismatched').get_labels()
    scriptsbase = "GLUE/mnli-mm"
    scriptformat = "txt"
    cutoff = 0.5 if (not args.nocut) else 0.0
    max_seq_l = 256
    batch_s = 500
    num_labels = [i for i in range(3)]
elif args.dataset == "qnli":
    # dataset['train'] = MNLIProcessor('mnli-m').get_examples('train')
    dataset['test'] = QNLIProcessor().get_examples('validation')
    class_labels = QNLIProcessor().get_labels()
    scriptsbase = "GLUE/qnli"
    scriptformat = "txt"
    cutoff = 0.5 if (not args.nocut) else 0.0
    max_seq_l = 300
    batch_s = 400
    num_labels = [i for i in range(2)]
elif args.dataset == "cola":
    # dataset['train'] = MNLIProcessor('mnli-m').get_examples('train')
    dataset['test'] = COLAProcessor().get_examples('validation')
    class_labels = QNLIProcessor().get_labels()
    scriptsbase = "GLUE/cola"
    scriptformat = "txt"
    cutoff = 0.5 if (not args.nocut) else 0.0
    max_seq_l = 64
    batch_s = 300
    num_labels = [i for i in range(2)]
elif args.dataset == "mrpc":
    # dataset['train'] = MNLIProcessor('mnli-m').get_examples('train')
    dataset['test'] = MRPCProcessor().get_examples('validation')
    class_labels = MRPCProcessor().get_labels()
    scriptsbase = "GLUE/mrpc"
    scriptformat = "txt"
    cutoff = 0.5 if (not args.nocut) else 0.0
    max_seq_l = 128
    batch_s = 300
    num_labels = [i for i in range(2)]
elif args.dataset == "qqp":
    # dataset['train'] = MNLIProcessor('mnli-m').get_examples('train')
    dataset['test'] = QQPProcessor().get_examples('validation')
    class_labels = QQPProcessor().get_labels()
    scriptsbase = "GLUE/qqp"
    scriptformat = "txt"
    cutoff = 0.5 if (not args.nocut) else 0.0
    max_seq_l = 256
    batch_s = 300
    num_labels = [i for i in range(2)]
elif args.dataset == "rte":
    # dataset['train'] = MNLIProcessor('mnli-m').get_examples('train')
    dataset['test'] = RTEProcessor().get_examples('validation')
    class_labels = RTEProcessor().get_labels()
    scriptsbase = "GLUE/rte"
    scriptformat = "txt"
    cutoff = 0.5 if (not args.nocut) else 0.0
    max_seq_l = 512
    batch_s = 100
    num_labels = [i for i in range(2)]
elif args.dataset == "stsb":
    # dataset['train'] = MNLIProcessor('mnli-m').get_examples('train')
    dataset['test'] = STSBProcessor().get_examples('validation')
    class_labels = STSBProcessor().get_labels()
    scriptsbase = "GLUE/stsb"
    scriptformat = "txt"
    cutoff = 0.5 if (not args.nocut) else 0.0
    max_seq_l = 128
    batch_s = 300
    num_labels = [i for i in range(2)]
else:
    raise NotImplementedError

mytemplate = ManualTemplate(tokenizer=tokenizer).from_file(f"{args.openprompt_path}/scripts/{scriptsbase}/manual_template.txt", choice=args.template_id)

if args.verbalizer == "ept":
    myverbalizer = EmbVerbalizer(tokenizer, model=plm, classes=class_labels, candidate_frac=cutoff, max_token_split=args.max_token_split, sumprob=args.sumprob, verbose=args.verbose).from_file(
        select_num=args.select, truncate=args.truncate, dataset_name=args.dataset, path=f'{args.dataset}_{args.model}_cos.pt', tmodel=args.model)
elif args.verbalizer == "manual":
    myverbalizer = ManualVerbalizer(tokenizer, classes=class_labels).from_file(f"{args.openprompt_path}/scripts/{scriptsbase}/manual_verbalizer.{scriptformat}")
elif args.verbalizer == "soft":
    raise NotImplementedError
elif args.verbalizer == "auto":
    raise NotImplementedError

prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False, plm_eval_mode=args.plm_eval_mode)
if use_cuda:
    prompt_model = prompt_model.cuda()

# zero-shot test
test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
                                   tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3,
                                   batch_size=batch_s, shuffle=False, teacher_forcing=False, predict_eos_token=False,
                                   truncate_method="tail")
allpreds = []
allprobs = []
alllabels = []
pbar = tqdm(test_dataloader)

all_stat = []

for step, inputs in enumerate(pbar):
    if use_cuda:
        inputs = inputs.cuda()
    stat = prompt_model(inputs)  # batch_size * num_class, 30 * 6
    # all_stat.append(stat)
    labels = inputs['label']
    alllabels.extend(labels.cpu().tolist())
    allpreds.extend(torch.argmax(stat, dim=-1).cpu().tolist())
    allprobs.append(torch.softmax(stat, dim=-1).cpu())
acc = sum([int(i == j) for i, j in zip(allpreds, alllabels)]) / len(allpreds)

clas_tag = True
try:
    matt_cor = matthews_corrcoef(alllabels, allpreds)
    print(classification_report(alllabels, allpreds, labels=num_labels, digits=4))
    report_dict = classification_report(alllabels, allpreds, labels=num_labels, digits=4, output_dict=True)
except:
    clas_tag = False

allscores = None
try:
    allprobs = torch.cat(allprobs, dim=0)
    allscores = (allprobs * torch.tensor(class_labels)).sum(dim=-1)
except Exception as e:
    print('Error for calculating pearson')

assert clas_tag or (not clas_tag and allscores is not None)

pearson = -10
if allscores is not None:
    pearson = pearsonr(alllabels, allscores)

content_write = "=" * 20 + "\n"
content_write += f"dataset {args.dataset}\t"
content_write += f"temp {args.template_id}\t"
content_write += f"seed {args.seed}\t"
content_write += f"verb {args.verbalizer}\t"
content_write += f"select {args.select}\t"
content_write += f"truncate {args.truncate}\t"
content_write += f"sumprob {args.sumprob}\t"
content_write += f"model {args.model_name_or_path}\t"
content_write += "\n"
if clas_tag:
    content_write += f"Acc: {acc}"
    content_write += "\n\n"
    content_write += f"Matthew's Corr: {matt_cor}"
    content_write += "\n\n"
    content_write += f"F1: {report_dict['1']['f1-score']}"
    content_write += "\n\n"
else:
    content_write += f"Pearson Coef: {pearson[0]}, p-value: {pearson[1]}"
    content_write += "\n\n"

print(content_write)

with open(f"{args.result_file}", "a") as fout:
    fout.write(content_write)
