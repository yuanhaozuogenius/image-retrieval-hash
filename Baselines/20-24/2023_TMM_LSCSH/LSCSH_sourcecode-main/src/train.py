import os, sys

SRC_DIR = os.path.abspath(os.path.dirname(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import datetime
import shutil

from Loss import Loss, HashCenterLoss
from common.utils import *
from common.logger import Logger
from dataloader.DataSet_loader import getDataLoader
from evaluate.measure_utils import *
from network import HashModel, CenterModel, AlexNetFc
from options import parser


class Engine(object):
    def __init__(self, option, state):
        self.option = option
        self.state = state
        self.class_num_dict = {'voc2012': 20, 'coco': 80, 'nuswide': 21}
        self.is_multi_label = option.data_name in ['voc2012', 'coco', 'nuswide']

    def useGPU(self, x):
        return x.cuda() if self.option.use_gpu and torch.cuda.is_available() else x

    def main(self):
        train_loader, test_loader, database_loader = getDataLoader(self.option)
        if self.option.model_type == 'resnet50':
            hash_model = HashModel(self.option)
        elif self.option.model_type == 'alexnet':
            hash_model = AlexNetFc(self.option)
        elif self.option.model_type == 'conformer':
            hash_model = Haformer(self.option)
        center_model = CenterModel(self.option)
        criterion = Loss(self.option, self.state)
        criterion_center = HashCenterLoss(self.option, self.state) if self.option.center_update else None

        optimizer_hash = torch.optim.Adam(hash_model.getConfig_params(), lr=self.option.lr, weight_decay=1e-5)
        optimizer_center = torch.optim.Adam(center_model.getConfig_params(),
                                            lr=self.option.lr_center) if self.option.center_update else None

        if self.option.resume:
            start_epoch = self.resume(hash_model, center_model, optimizer_hash, optimizer_center)
        else:
            start_epoch = -1

        if self.option.use_gpu and torch.cuda.is_available():
            hash_model = torch.nn.DataParallel(hash_model).cuda()
            center_model = torch.nn.DataParallel(center_model).cuda()
            criterion = criterion.cuda()
            if self.option.center_update:
                criterion_center = criterion_center.cuda()

        self.run_epoch(hash_model, center_model, criterion, criterion_center, optimizer_hash, optimizer_center,
                       train_loader, test_loader, database_loader, start_epoch)

    def run_epoch(self, hash_model, center_model, criterion, criterion_center, optimizer_hash, optimizer_center,
                  train_loader, test_loader, database_loader, start_epoch):
        centerWeight_train = self.initCenterWeight(train_loader)
        self.state.update({
            'best_MAP': 0.0, 'best_epoch': 0,
            'Database_hashpool_path': None, 'Testbase_hashpool_path': None,
            'Trainbase_hashpool_path': None, 'final_result': None, 'filename_previous_best': None,
            'interClass_loss': []
        })
        Logger.divider("start training..")

        for epoch in range(start_epoch + 1, self.option.epochs):
            lr = self.adjust_learning_rate(optimizer_hash, epoch)
            lr_center = self.adjust_learning_rate(optimizer_center, epoch, 'centerNet')
            Logger.divider(f"epoch[{epoch}] lr:{lr} lr_center:{lr_center}")
            if optimizer_center:
                optimizer_center.zero_grad()
            self.on_start_epoch(epoch)

            hashCenter_pre, word_embedding = self.forward_hashCenter(center_model, train_loader)
            self.state['hash_center_pre'] = hashCenter_pre

            loss_epoch = self.train(hash_model, train_loader, criterion, hashCenter_pre, optimizer_hash,
                                    centerWeight_train, epoch)
            Logger.info(f"\tEpoch : {epoch}, Mean Loss {np.mean(loss_epoch)}\n")

            is_best = False
            if epoch >= self.option.start_test_epoch:
                (Precision, Recall1, MAP1), (MAP2, Recall2, Precision2) = self.test(
                    hash_model, train_loader, test_loader, database_loader, centerWeight_train, epoch)
                Logger.info(f"epoch {epoch} Result：{(Precision, Recall1, MAP1), (MAP2, Recall2, Precision2)}")

                if self.option.center_update:
                    self.updateCenter(criterion_center, word_embedding, centerWeight_train, hashCenter_pre,
                                      optimizer_center)

                self.saveStatus(epoch, centerWeight_train, hashCenter_pre, MAP2,
                                ((Precision, Recall1, MAP1), (MAP2, Recall2, Precision2)))
                is_best = MAP2 >= self.state['best_MAP']
                Logger.info(
                    f"MAP epoch {epoch}\tMAP_best {self.state['best_MAP']}\tIs_best {is_best}\tBest epoch {self.state['best_epoch']}")

            self.on_end_epoch(self.option, self.state, epoch, hash_model, center_model, is_best,
                              optimizer_hash, optimizer_center)

        Logger.info("start drawing ...")
        if self.state["final_result"] is not None:
            Logger.info(f"Hash Pool Radius :{self.option.R}\n"
                        f"MAP1 :{self.state['final_result'][1][0]:.4f}\t Recall1 {self.state['final_result'][1][1]:.4f}\t"
                        f"Precision1 {self.state['final_result'][1][2]:.4f}\t MAP2 {self.state['final_result'][0][2]:.4f}\t"
                        f"Recall2 {self.state['final_result'][0][1]:.4f} \t Precision2 {self.state['final_result'][0][0]:.4f} ")
        else:
            Logger.info(
                "⚠️ [Warning] No MAP result was recorded. Possibly due to too few epochs or no valid evaluation point.")

    def updateCenter(self, criterion_center, word_embedding, weightCenter, hashCenter_pre, optimizer):
        optimizer.zero_grad()
        Logger.info("\t<==update center==>")

        loss = criterion_center(word_embedding, hashCenter_pre,
                                weightCenter, getTrainbaseHashPoolPath(self.option, self.state))
        loss.backward()
        optimizer.step()

    def on_start_epoch(self, epoch):
        self.state['epoch'] = epoch

        pass

    def on_end_epoch(self, option, state, epoch, model_hash, model_center, is_best, optimizer_hash, optimizer_center):

        model_dict = {
            'epoch': epoch,
            'model_hash_dict': model_hash.module.state_dict() if option.use_gpu and torch.cuda.is_available() else model_hash.state_dict(),
            'model_center_dict': model_center.module.state_dict() if option.use_gpu and torch.cuda.is_available() else model_center.state_dict(),
            'optimizer_hash_dict': optimizer_hash.state_dict(),
            'optimizer_center_dict': optimizer_center.state_dict(),
            'best_MAP': state['best_MAP']
        }
        self.save_checkpoint(option, state, model_dict, is_best)

    pass

    def train(self, model, train_loader, criterion, hash_center_pre, optimizer, centerWeight, epoch):
        model.train()

        loss_epoch = []

        train_loader = tqdm(train_loader, desc="Epoch [" + str(epoch) + "]==>Training:")
        for i, (input, target) in enumerate(train_loader):
            images = input[0]
            hash_code = model(images)
            centerWeight_batch = self.useGPU(
                torch.tensor(centerWeight[i * self.option.batch_size:(i + 1) * self.option.batch_size],
                             requires_grad=True))

            hash_centroid = self.getHashCenters(target, hash_center_pre.detach(), centerWeight_batch)

            if not self.option.fixed_weight:
                optimizer.zero_grad()
                centerWeight_batch.retain_grad()
                loss = criterion(hash_code.detach(), hash_centroid, hash_center_pre.detach(), centerWeight_batch,
                                 target)
                loss.backward(retain_graph=True)
                weight_grad = torch.where(torch.isnan(centerWeight_batch.grad), self.useGPU(torch.tensor(0.)),
                                          centerWeight_batch.grad)
                centerWeight_batch = centerWeight_batch - self.option.eta * weight_grad
                centerWeight_batch = self.simplexPro(centerWeight_batch)

                centerWeight[
                i * self.option.batch_size:(i + 1) * self.option.batch_size] = centerWeight_batch.cpu().detach().numpy()
            optimizer.zero_grad()
            loss = criterion(hash_code, hash_centroid.detach(), hash_center_pre.detach(), centerWeight_batch.detach(),
                             target)
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.data.cpu().numpy())
        return loss_epoch

    def test(self, model, train_loader, test_loader, database_loader, centerWeight, epoch):
        model.eval()
        self.predict_hash_code(model, database_loader, centerWeight, epoch, database_type="database")
        self.predict_hash_code(model, test_loader, centerWeight, epoch, database_type="testbase")
        self.predict_hash_code(model, train_loader, centerWeight, epoch, database_type="trainbase")

        database_hashcode, database_labels = loadHashPool(self.option, self.state,
                                                                getDatabaseHashPoolPath(self.option, self.state),
                                                                'database')
        testbase_hashcode, testbase_labels = loadHashPool(self.option, self.state,
                                                                getTestbaseHashPoolPath(self.option, self.state))

        database_hashcode_numpy = database_hashcode.detach().cpu().numpy().astype('float32')
        database_labels_numpy = database_labels.detach().cpu().numpy().astype('int8')
        testbase_hashcode_numpy = testbase_hashcode.detach().cpu().numpy().astype('float32')
        testbase_labels_numpy = testbase_labels.detach().cpu().numpy().astype('int8')

        del database_hashcode, database_labels, testbase_hashcode, testbase_labels
        Logger.info("===> start calculate MAP!\n")

        Precision, Recall1, MAP1 = get_precision_recall_by_Hamming_Radius_optimized(
            database_hashcode_numpy, database_labels_numpy, testbase_hashcode_numpy, testbase_labels_numpy,
            fine_sign=True)

        MAP2, Recall2, P = mean_average_precision(database_hashcode_numpy, testbase_hashcode_numpy,
                                                  database_labels_numpy, testbase_labels_numpy, self.option)
        del database_hashcode_numpy, testbase_hashcode_numpy, database_labels_numpy, testbase_labels_numpy
        return (Precision, Recall1, MAP1), (MAP2, Recall2, P)

    def predict_hash_code(self, model, data_loader, centerWeight, epoch, database_type):
        model.eval()
        path_func = {
            'database': getDatabaseHashPoolPath,
            'testbase': getTestbaseHashPoolPath,
            'trainbase': getTrainbaseHashPoolPath,
        }
        filename = path_func[database_type](self.option, self.state)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'ab') as file_path:
            for i, (input, target) in enumerate(tqdm(data_loader, desc=f"epoch[{epoch}]==>{database_type}==>Testing:")):
                images = input[0]
                hash_code = model(images)
                if database_type == 'trainbase':
                    centerWeight_batch = self.useGPU(
                        torch.tensor(centerWeight[i * self.option.batch_size:(i + 1) * self.option.batch_size]))
                    center = self.getHashCenters(target, self.state['hash_center_pre'], centerWeight_batch)
                    save_obj = {
                        'output': hash_code.cpu(),
                        'target': target.cpu(),
                        'center': center.cpu(),
                        'weight': centerWeight_batch.cpu()
                    }
                else:
                    save_obj = {'output': hash_code.cpu(), 'target': target.cpu()}
                pickle.dump(save_obj, file_path)

    def adjust_learning_rate(self, optimizer, epoch, type='hashNet'):

        if type == 'hashNet':
            lr = option.lr * (0.7 ** (epoch // 10))
            optimizer.param_groups[0]['lr'] = option.multi_lr * lr
            optimizer.param_groups[1]['lr'] = lr
        elif type == 'centerNet' and self.option.center_update:
            lr = option.lr_center * (0.7 ** (epoch // 10))
            optimizer.param_groups[0]['lr'] = lr
        elif type == 'centerNet' and not self.option.center_update:
            lr = 0
        return lr

    def getHashCenters(self, labels, hash_centers, center_weight):

        if self.is_multi_label:
            hash_centers = self.Hash_center_multilables(labels, hash_centers, center_weight=center_weight)
        else:
            hash_label = (labels == 1).nonzero()[:, 1]
            hash_centers = hash_centers[hash_label]
        return hash_centers

    def Hash_center_multilables(self, labels,
                                Hash_center_pre,
                                center_weight):
        hash_centers = self.useGPU(torch.FloatTensor(torch.FloatStorage()))
        for (i, label) in enumerate(labels):
            one_labels = (label == 1).nonzero()
            one_labels = one_labels.squeeze(1)
            Centers = Hash_center_pre[one_labels][:]
            center_weight_one = center_weight[i][one_labels]
            center_mean = torch.sum(Centers * center_weight_one.view(-1, 1), dim=0)
            hash_centers = torch.cat((hash_centers, center_mean.view(1, -1)), dim=0)
        return hash_centers

    def forward_hashCenter(self, centerModel, loader):

        for i, (input, target) in enumerate(loader):
            word_embedding = self.useGPU(input[2][0])
            hashCenter_pre = centerModel(word_embedding)
            return hashCenter_pre

    def initCenterWeight(self, data_loader):

        if self.option.resume:
            if os.path.exists(self.option.resume_weight_path):
                return np.load(self.option.resume_weight_path)
            else:
                Logger.info(" lose weight path !")
                sys.exit()
        data_name = self.option.data_name
        folder = '../data/' + data_name + '/'
        file_path = os.path.join(folder, data_name + '_initial_weight.npy')

        # 自动创建文件夹
        if not os.path.exists(folder):
            os.makedirs(folder)
        if os.path.exists(file_path):
            self.state['centerWeight_path'] = file_path
            return np.load(file_path)

        Logger.info("init center weight..")
        all_weight = None
        data_loader = tqdm(data_loader, desc="[init center weight]")
        for i, (input, target) in enumerate(data_loader):
            center_num = torch.sum(target > 0, dim=1)
            target[target <= 0] = 0.
            centerWeight = target.float() / center_num.view(-1, 1).float()

            if i == 0:
                all_weight = centerWeight.data.cpu().float()
            else:
                all_weight = torch.cat((all_weight, centerWeight.data.cpu().float()), 0)
        np.save(file_path, all_weight.cpu().numpy())
        return all_weight.cpu().numpy()

    def simplexPro(self, weightCenter_pre):
        for i in range(len(weightCenter_pre)):
            weight = weightCenter_pre[i]
            index = (weight > 0).nonzero().squeeze(1)
            X = torch.sort(weight[index])[0]
            Y = self.useGPU(torch.arange(1, len(index) + 1, 1).float())

            R = X + (1. / Y) * (1. - torch.cumsum(X, dim=0))
            Y[R <= 0.] = 0.
            rou = torch.max(Y).int()
            lambda_ = (1 / rou.float()) * (1 - torch.cumsum(X, dim=0)[rou - 1].float())
            temp = weight[index] + lambda_
            temp[temp < 0] = 0.
            weight[index] = temp
        return weightCenter_pre

    def pickleDump(self, content, filePath):
        pickle.dump(content, filePath)

    def save_checkpoint(self, option, state, model_dict, is_best, filename='checkpoint.pth.tar'):
        save_model_path = '../result/' + option.data_name + '/model'
        if option.data_name is not None:
            filename_ = filename
            filename = os.path.join(save_model_path, filename_)
            if not os.path.exists(save_model_path):
                os.makedirs(save_model_path)
        Logger.info('save model {filename}\n'.format(filename=filename))
        torch.save(model_dict, filename)
        if is_best:
            filename_best = 'model_best.pth.tar'
            if save_model_path is not None:
                filename_best = os.path.join(save_model_path, filename_best)
            shutil.copyfile(filename, filename_best)
            if save_model_path is not None:
                if state['filename_previous_best'] is not None and os.path.exists(state['filename_previous_best']):
                    os.remove(state['filename_previous_best'])
                filename_best = os.path.join(save_model_path,
                                             'model_best_{score:.4f}.pth.tar'.format(score=model_dict['best_MAP']))
                shutil.copyfile(filename, filename_best)
                state['filename_previous_best'] = filename_best

    def saveStatus(self, epoch, centerWeight_train, hashCenter_pre, MAP, result_all=None):

        np.save('../result/' + self.option.data_name + '/centers.npy', hashCenter_pre.detach().cpu().numpy())
        if MAP >= self.state['best_MAP']:
            if self.state['Database_hashpool_path'] is not None and os.path.exists(
                    self.state['Database_hashpool_path']):
                os.remove(self.state['Database_hashpool_path'])
            if self.state['Testbase_hashpool_path'] is not None and os.path.exists(
                    self.state['Testbase_hashpool_path']):
                os.remove(self.state['Testbase_hashpool_path'])
            if self.state['Trainbase_hashpool_path'] is not None and os.path.exists(
                    self.state['Trainbase_hashpool_path']):
                os.remove(self.state['Trainbase_hashpool_path'])
            self.state['Database_hashpool_path'] = getDatabaseHashPoolPath(self.option, self.state)
            self.state['Testbase_hashpool_path'] = getTestbaseHashPoolPath(self.option, self.state)
            self.state['Trainbase_hashpool_path'] = getTrainbaseHashPoolPath(self.option, self.state)
            self.state['best_MAP'] = MAP
            self.state['best_epoch'] = epoch
            self.state['final_result'] = result_all
            np.save(getWeightBestPath(self.option, self.state), centerWeight_train)
            pass
        elif epoch >= self.option.epochs - 1:
            np.save('../result/' + self.option.data_name + '/finalweight.npy', centerWeight_train)
            pass
        else:
            if os.path.exists(getDatabaseHashPoolPath(self.option, self.state)):
                os.remove(getDatabaseHashPoolPath(self.option, self.state))
            if os.path.exists(getTestbaseHashPoolPath(self.option, self.state)):
                os.remove(getTestbaseHashPoolPath(self.option, self.state))
            if os.path.exists(getTrainbaseHashPoolPath(self.option, self.state)):
                os.remove(getTrainbaseHashPoolPath(self.option, self.state))
        pass

    def resume(self, model_hash, model_center, optimizer_hash, optimizer_center):
        path_checkpoint = option.resume_path
        if option.resume and os.path.exists(path_checkpoint):
            checkpoint = torch.load(path_checkpoint)
            model_hash.load_state_dict(checkpoint['model_hash_dict'])
            model_center.load_state_dict(checkpoint['model_center_dict'])
            optimizer_hash.load_state_dict(checkpoint['optimizer_hash_dict'])
            optimizer_center.load_state_dict(checkpoint['optimizer_center_dict'])
            if option.use_gpu and torch.cuda.is_available():
                for state in optimizer_hash.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()
                for state in optimizer_center.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()

            return checkpoint['epoch']
        else:
            Logger.info("checkpoint file not exist!  ")
            sys.exit()
        pass


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    Logger.info("\t\tstart program\t\t")
    option = parser.parse_args()
    Logger.divider("print option")
    for k, v in vars(option).items():
        Logger.info('\t{}: {}'.format(k, v))
    state = {'start_time': start_time}
    engine = Engine(option=option, state=state)
    engine.main()
    end_time = datetime.datetime.now()
    Logger.divider("END {}".format(Logger.getTimeStr(end_time)))
