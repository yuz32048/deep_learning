import torch
import numpy as np
from tqdm import tqdm
import random
from transformers import AutoConfig
from transformers import AutoModelWithLMHead

from src.utils import *
from src.dataloader import *
from src.trainer import *
from src.config import *

import matplotlib.pyplot as plt

class BertTagger(nn.Module):
    def __init__(self, hidden_dim, output_dim, model_name):
        super(BertTagger, self).__init__()
        # TODO:
        # （1）利用AutoConfig.from_pretrained定义config
        # （2）利用AutoModelWithLMHead.from_pretrained定义模型，注意要传入刚才的config
        # （3）定义一个线性层用于分类预测
        # 提示：参考文档https://huggingface.co/bert-base-cased
        config = AutoConfig.from_pretrained(model_name)
        self.bert_model = AutoModelWithLMHead.from_pretrained(model_name, config=config)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        # TODO: 
        # （1）把X输入bert_model得到hidden_states；
        # （2）提取其中属于最后一个transformer layer的hidden_states作为最终特征；
        # （3）最后把特征输入线性层完成预测。
        # 提示：需要用到output_hidden_states参数，并参考以下文档
        # https://huggingface.co/docs/transformers/v4.21.2/en/model_doc/bert#transformers.BertLMHeadModel
        features = self.bert_model(input_ids=X, output_hidden_states=True).hidden_states[-1]
        logits = self.classifier(features)
        return logits
        

def main(params):
    # ===========================================================================
    # Using Fixed Random Seed
    if params.seed:
        random.seed(params.seed)
        np.random.seed(params.seed)
        torch.manual_seed(params.seed)
        torch.cuda.manual_seed(params.seed)
        torch.backends.cudnn.deterministic = True
    # Initialize Experiment
    logger = init_experiment(params, logger_filename=params.logger_filename)
    logger.info(params.__dict__)
    # Set domain name
    domain_name = os.path.basename(params.data_path[0])
    if domain_name=='':
        # Remove the final char '\' in the path
        domain_name = os.path.basename(params.data_path[0][:-1])
    # Generate Dataloader 
    ner_dataloader = NER_dataloader(data_path=params.data_path,
                                    domain_name=domain_name,
                                    batch_size=params.batch_size, 
                                    entity_list=params.entity_list)
    dataloader_train, dataloader_dev, dataloader_test = ner_dataloader.get_dataloader()
    label_list = ner_dataloader.label_list
    entity_list = ner_dataloader.entity_list

    # NER Tagger
    if params.model_name in ['bert-base-cased','roberta-base']:
        model = BertTagger(hidden_dim=params.hidden_dim,
                            output_dim=len(label_list), 
                            model_name=params.model_name)
    else:
        raise Exception('model name %s is invalid'%params.model_name)
    model.cuda()
    trainer = BaseTrainer(params, model, entity_list, label_list)

    # ===========================================================================
    # Start training
    logger.info("Training ...")
    # Init variables
    no_improvement_num = 0
    best_f1 = 0
    step = 0

    logger.info("Initial lr is %s"%( str(trainer.scheduler.get_last_lr())))

    # 记录训练过程中的指标
    loss_list_train = []
    f1_list_dev = []

    for e in range(1,params.training_epochs+1):

        logger.info("============== epoch %d ==============" % e)
        loss_list = []
        mean_loss = 0.0
        total_cnt = 0
        correct_cnt = 0

        pbar = tqdm(dataloader_train, total=len(dataloader_train))
        for X, y in pbar:

            # Update the step count
            step += 1
            X, y = X.cuda(), y.cuda()
            trainer.batch_forward(X)

            # Record training accuracy
            correct_cnt += int(torch.sum(torch.eq(torch.max(trainer.logits,dim=2)[1],y).float()).item())
            total_cnt += trainer.logits.size(0)*trainer.logits.size(1)

            # Compute CE loss
            trainer.batch_loss(y)
            loss = trainer.batch_backward()
            loss_list.append(loss)
            mean_loss = np.mean(loss_list)

            pbar.set_description("Epoch %d, Step %d: Loss=%.4f, Training_acc=%.2f%%"%(
                    e, step, mean_loss, correct_cnt/total_cnt*100
            ))

        # 记录每轮训练的损失
        loss_list_train.append(mean_loss)

        if params.info_per_epochs>0 and e%params.info_per_epochs==0:
            logger.info("Epoch %d, Step %d: Loss=%.4f, Training_acc=%.2f%%"%(
                e, step, mean_loss, correct_cnt/total_cnt*100
            ))

        # Update learning rate
        if trainer.scheduler != None:
            old_lr = trainer.scheduler.get_last_lr()
            trainer.scheduler.step()
            new_lr = trainer.scheduler.get_last_lr()
            if old_lr != new_lr:
                logger.info("Epoch %d, Step %d: lr is %s"%(
                    e, step, str(new_lr)
                ))

        # Save checkpoint 
        if params.save_per_epochs!=0 and e%params.save_per_epochs==0:
            trainer.save_model("best_finetune_domain_%s_epoch_%d.pth"%(domain_name, e), path=params.dump_path)
        
        # For evaluation
        if e%params.evaluate_interval==0:

            f1_dev, f1_dev_each_class = trainer.evaluate(dataloader_dev, each_class=True)
            logger.info("Epoch %d, Step %d: Dev_f1=%.4f, Dev_f1_each_class=%s" % (
                e, step, f1_dev, str(f1_dev_each_class)
            ))

            # 纪记录每次评估后的f1
            f1_list_dev.append(f1_dev)

            if f1_dev > best_f1:
                logger.info("Find better model!!")
                best_f1 = f1_dev
                no_improvement_num = 0
                trainer.save_model("best_finetune_domain_%s.pth"%domain_name, path=params.dump_path)
            else:
                no_improvement_num += 1
                logger.info("No better model is found (%d/%d)" % (no_improvement_num, params.early_stop))
            
            if no_improvement_num >= params.early_stop:
                logger.info("Stop training because no better model is found!!!")
                break

    logger.info("Finish training ...")

    # ===========================================================================
    # testing
    logger.info("Testing...")
    # trainer.load_model("best_finetune_domain_%s_step_2000.pth"%domain_name, path=params.dump_path)
    trainer.load_model("best_finetune_domain_%s.pth"%domain_name, path=params.dump_path)
    trainer.model.cuda()
    f1_test, f1_score_dict = trainer.evaluate(dataloader_test, each_class=True)
    logger.info("Final Result: Evaluate on Test Set. F1: %.4f." % (f1_test))
    f1_score_dict = sorted(f1_score_dict.items(),key=lambda x:x[0])
    logger.info("F1_list: %s" % (f1_score_dict))
    logger.info("Finish testing ...")

    # 可视化训练过程中的指标
    # (1) 损失变化图
    plt.figure()
    plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
    plt.title("损失变化图")
    plt.xlabel("训练轮次epoch")
    plt.ylabel("损失Loss")
    plt.plot(loss_list_train, color = "black")
    plt.savefig("./loss.png")
    plt.show()

    # (2) F1变化图
    plt.figure()
    plt.title("F1变化图")
    plt.ylabel("F1")
    plt.xlabel("epoch")
    plt.plot(f1_list_dev, color = "red")
    plt.savefig("./f1_dev.png")
    plt.show()

if __name__ == "__main__":
    params = get_params()
    main(params)
