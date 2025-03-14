# %%
import pandas as pd
import openpyxl
import h5py
import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import sys
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
# %% md
## Experiment Hyperparameters
# %%
is_local = False

# Experiment
seed = 1000 if is_local else int(sys.argv[-2])
torch.manual_seed(seed)
image_size = 256

# Data: which wavenumbers are even allowed to be considered?
wv_start = 0
wv_end = 965

# Data loading
test_set_fraction = 0.2
val_set_fraction = 0.2
batch_size = 64
patch_dim = 1
use_augmentation = False
samples_to_train = 10000

# dimensionality reduction parameters
r_method = 'fixed'  # 'fixed' or 'pca'
reduce_dim = 4 if is_local else int(sys.argv[-1])
channels_used = np.s_[..., :]  # used only when r_method = 'fixed'
print(channels_used)


# %%
def csf_fp(filepath):
    return filepath.replace('D:/datasets', 'D:/datasets' if is_local else './')


master = pd.read_excel(csf_fp(rf'D:/datasets/pcuk2023_ftir_whole_core/master_sheet.xlsx'))
slide = master['slide'].to_numpy()
patient_id = master['patient_id'].to_numpy()
hdf5_filepaths = np.array([csf_fp(fp) for fp in master['hdf5_filepath']])
annotation_filepaths = np.array([csf_fp(fp) for fp in master['annotation_filepath']])
mask_filepaths = np.array([csf_fp(fp) for fp in master['mask_filepath']])
wavenumbers = np.load(csf_fp(f'D:/datasets/pcuk2023_ftir_whole_core/wavenumbers.npy'))[wv_start:wv_end]
wavenumbers_used = wavenumbers[channels_used]

annotation_class_colors = np.array([[0, 255, 0], [128, 0, 128], [255, 0, 255], [0, 0, 255], [255, 165, 0], [255, 0, 0]])
annotation_class_names = np.array(['epithelium_n', 'stroma_n', 'epithelium_c', 'stroma_c', 'corpora_amylacea', 'blood'])
n_classes = len(annotation_class_names)
print(f"Loaded {len(slide)} cores")
print(f"Using {len(wavenumbers_used)}/{len(wavenumbers)} wavenumbers")
# %% md
## Define Datasets, Dataloaders
# %%
unique_pids = np.unique(patient_id)
pids_trainval, pids_test, _, _ = train_test_split(
    unique_pids, np.zeros_like(unique_pids), test_size=test_set_fraction, random_state=seed)
pids_train, pids_val, _, _ = train_test_split(
    pids_trainval, np.zeros_like(pids_trainval), test_size=(val_set_fraction / (1 - test_set_fraction)),
    random_state=seed)
where_train = np.where(np.isin(patient_id, pids_train))
where_val = np.where(np.isin(patient_id, pids_val))
where_test = np.where(np.isin(patient_id, pids_test))
print(
    f"Patients per data split:\n\tTRAIN: {len(where_train[0])}\n\tVAL: {len(where_val[0])}\n\tTEST: {len(where_test[0])}")


# %%
class ftir_patching_dataset(torch.utils.data.Dataset):
    def __init__(self, hdf5_filepaths, mask_filepaths, annotation_filepaths, channels_use,
                 patch_dim=25, augment=True, ):

        # Define data paths
        self.hdf5_filepaths = hdf5_filepaths
        self.mask_filepaths = mask_filepaths
        self.annotation_filepaths = annotation_filepaths
        self.augment = augment

        # patch dimensions
        self.patch_dim = patch_dim
        self.patch_minus = patch_dim // 2;
        self.patch_plus = 1 + (patch_dim // 2)
        self.channels = channels_use

        # class data
        self.annotation_class_colors = annotation_class_colors
        self.annotation_class_names = annotation_class_names
        self.total_sampled = torch.zeros(len(self.annotation_class_colors))

        # define data augmentation pipeline
        self.transforms = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
        ])

        # Open every core hdf5 file
        self.open()

    def __len__(self):
        return self.total_pixels

    def __getitem__(self, idx):
        # get patch data
        row = self.rows[idx]
        col = self.cols[idx]
        cidx = self.cidxs[idx]
        label = self.tissue_classes[idx]
        self.total_sampled[label] += 1

        # Are dimensions of patch okay
        idx_u = row - self.patch_minus
        idx_d = row + self.patch_plus
        idx_l = col - self.patch_minus
        idx_r = col + self.patch_plus
        pad_u = max(-idx_u, 0);
        idx_u = max(idx_u, 0)
        pad_d = max(idx_d - image_size, 0);
        idx_d = min(idx_d, image_size)
        pad_l = max(-idx_l, 0);
        idx_l = max(idx_l, 0)
        pad_r = max(idx_r - image_size, 0);
        idx_r = min(idx_r, image_size)

        # get patch
        patch = torch.from_numpy(
            self.hdf5_files[cidx]['spectra'][idx_u:idx_d, idx_l:idx_r, *self.channels],
        ).permute(2, 0, 1)
        patch *= torch.from_numpy(
            self.hdf5_files[cidx]['mask'][idx_u:idx_d, idx_l:idx_r, ],
        ).unsqueeze(0)

        # pad patch
        patch = torch.nn.functional.pad(patch, (pad_l, pad_r, pad_u, pad_d, 0, 0))

        if self.augment:
            patch = self.transforms(patch)
        return patch, label

    # split annotations from H x W x 3 to C x H x W, one/zerohot along C dimension
    def split_annotations(self, annotations_img):
        split = torch.zeros((len(self.annotation_class_colors), *annotations_img.shape[:-1]))
        for c, col in enumerate(annotation_class_colors):
            split[c, :, :] = torch.from_numpy(np.all(annotations_img == self.annotation_class_colors[c], axis=-1))
        return split

    # open every file
    def open(self):
        self.hdf5_files = []
        self.tissue_classes = []
        self.rows = []
        self.cols = []
        self.cidxs = []

        # for every core in dataset,
        for cidx in range(0, len(self.hdf5_filepaths)):
            # open annotations and remove edges and non-tissue px
            annotation = self.split_annotations(cv2.imread(self.annotation_filepaths[cidx])[:, :, ::-1])
            mask = torch.from_numpy(cv2.imread(self.mask_filepaths[cidx])[:, :, 1]) / 255
            annotation *= mask
            # for every class,
            for cls in range(len(annotation_class_names)):
                # get location of annotations, append to lists
                r, c = torch.where(annotation[cls])
                num_cls = annotation[cls].sum().int().item()
                self.tissue_classes.extend([cls, ] * num_cls)
                self.cidxs.extend([cidx, ] * num_cls)
                self.rows.extend(r)
                self.cols.extend(c)
            # add open hdf5 file to list
            self.hdf5_files.append(h5py.File(self.hdf5_filepaths[cidx], 'r'))

        # construct data tensors
        self.rows = torch.Tensor(self.rows).int()
        self.cols = torch.Tensor(self.cols).int()
        self.tissue_classes = torch.Tensor(self.tissue_classes).long()
        self.cidxs = torch.Tensor(self.cidxs).int()
        self.total_pixels = len(self.cidxs)

    # close every open hdf5 file
    def close(self):
        for cidx in range(len(self.hdf5_files)):
            self.hdf5_files[cidx].close()
        self.hdf5_files = []
        self.tissue_classes = []
        self.xs = []
        self.ys = []


# %%
dataset_train = ftir_patching_dataset(
    hdf5_filepaths[where_train], mask_filepaths[where_train], annotation_filepaths[where_train], channels_used,
    patch_dim=patch_dim, augment=use_augmentation,
)
dataset_val = ftir_patching_dataset(
    hdf5_filepaths[where_val], mask_filepaths[where_val], annotation_filepaths[where_val], channels_used,
    patch_dim=patch_dim, augment=False,
)
dataset_test = ftir_patching_dataset(
    hdf5_filepaths[where_test], mask_filepaths[where_test], annotation_filepaths[where_test], channels_used,
    patch_dim=patch_dim, augment=False,
)

# Instiantiate data loaders
_, class_counts = np.unique(dataset_train.tissue_classes, return_counts=True)
class_weights = 1 / class_counts
class_weights = class_weights[dataset_train.tissue_classes]
train_sampler = torch.utils.data.WeightedRandomSampler(class_weights, len(class_weights), replacement=True)

_, class_counts = np.unique(dataset_val.tissue_classes, return_counts=True)
class_weights = 1 / class_counts
class_weights = class_weights[dataset_val.tissue_classes]
val_sampler = torch.utils.data.WeightedRandomSampler(class_weights, len(class_weights), replacement=True)

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, sampler=train_sampler, drop_last=True)
val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, sampler=val_sampler, drop_last=True)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, drop_last=True)
print(f"loader sizes:\n\ttrain: {len(train_loader)}\n\tval: {len(val_loader)}\n\ttest: {len(test_loader)}")
# %% md
## Sample some data
# %%
train_data = []
train_labels = []
for iter in range(0, (samples_to_train // batch_size) + 1):
    for bidx, (data, label) in enumerate(train_loader):
        print(f"{bidx}/{(samples_to_train // batch_size)}", end="\r")
        train_data.append(data.squeeze().cpu().numpy())
        train_labels.append(label.squeeze().cpu().numpy())

        if iter * len(train_loader) + bidx * batch_size > samples_to_train:
            end = True
            break
    if end:
        break
train_data = np.concatenate(train_data, axis=0)
train_labels = np.concatenate(train_labels, axis=0)
# %% md
## Define, train Model
# %%
if r_method == 'pca':
    svm_model = svm_model = Pipeline([
        ("pca", PCA(n_components=reduce_dim)),
        ("normalise", StandardScaler(),),
        ("kernel_map", Nystroem(kernel='rbf', gamma=1e-4, n_components=500),),
        ("svm", LinearSVC(C=1.0, )),
    ])
    rf_model = Pipeline([
        ("pca", PCA(n_components=reduce_dim)),
        ("normalise", StandardScaler(),),
        ("randomforest", RandomForestClassifier(n_estimators=500, min_samples_leaf=10))
    ])
elif r_method == 'fixed':
    svm_model = Pipeline([
        ("normalise", StandardScaler(),),
        ("kernel_map", Nystroem(kernel='rbf', gamma=1e-4, n_components=500),),
        ("svm", LinearSVC(C=1.0, )),
    ])
    rf_model = Pipeline([
        ("normalise", StandardScaler(),),
        ("randomforest", RandomForestClassifier(n_estimators=500, min_samples_leaf=10))
    ])

svm_model.fit(train_data, train_labels)
rf_model.fit(train_data, train_labels)
print(f"svm accuracy on the train data: {accuracy_score(svm_model.predict(train_data), train_labels)}")
print(f"rf accuracy on the train data: {accuracy_score(rf_model.predict(train_data), train_labels)}")
# %% md
## Test Model
# %%
test_preds_svm, test_preds_rf, test_targets = [], [], []

for bidx, (data, label) in enumerate(test_loader):
    print(f"{bidx}/{(len(test_loader))}", end="\r")
    data = data.squeeze().cpu().numpy()
    label = label.squeeze().cpu().numpy()
    pred_svm = svm_model.predict(data)
    pred_rf = rf_model.predict(data)

    test_preds_svm.extend(pred_svm)
    test_preds_rf.extend(pred_rf)
    test_targets.extend(label)

test_targets = np.array(test_targets);
test_preds_svm = np.array(test_preds_svm);
test_preds_rf = np.array(test_preds_rf)
test_acc_svm = accuracy_score(test_targets, test_preds_svm)
test_f1m_svm = f1_score(test_targets, test_preds_svm, average='macro')
test_f1_svm = f1_score(test_targets, test_preds_svm, average=None)
test_acc_rf = accuracy_score(test_targets, test_preds_rf)
test_f1m_rf = f1_score(test_targets, test_preds_rf, average='macro')
test_f1_rf = f1_score(test_targets, test_preds_rf, average=None)
print("Metrics on entire testing set:")
print(f"TEST ---- | RANDOMFOREST | OA: {test_acc_rf:.4f} | f1: {test_f1m_rf:.4f}")
for cls_idx, f1 in enumerate(test_f1_rf):
    print(f"{annotation_class_names[cls_idx]}{(20 - len(annotation_class_names[cls_idx])) * ' '} : {f1:.4f}")

print(f"TEST ---- | SVM | OA: {test_acc_svm:.4f} | f1: {test_f1m_svm:.4f}")
for cls_idx, f1 in enumerate(test_f1_svm):
    print(f"{annotation_class_names[cls_idx]}{(20 - len(annotation_class_names[cls_idx])) * ' '} : {f1:.4f}")

print("Total samples loaded for each class during TESTING")
for cls_idx, samples_loaded in enumerate(dataset_test.total_sampled.numpy()):
    print(
        f"{annotation_class_names[cls_idx]}{(20 - len(annotation_class_names[cls_idx])) * ' '}:    {int(samples_loaded)}")
# %% md
## Finish experiment
# %%
# Read existing results file
if not is_local:
    if os.path.isfile('results_svm.txt'):
        f = open('results_svm.txt', 'r')
        lines = f.readlines()
        f.close()
    else:
        lines = [x + ', \n' for x in ['seed', *annotation_class_names, 'overall_acc', 'macro_f1']]

    # Process files
    lines[0] = lines[0].replace('\n', str(seed) + ', \n')
    for cls in range(n_classes):
        lines[cls + 1] = lines[cls + 1].replace('\n', str(test_f1_svm[cls]) + ', \n')
    lines[n_classes + 1] = lines[n_classes + 1].replace('\n', str(test_acc_svm) + ', \n')
    lines[n_classes + 2] = lines[n_classes + 2].replace('\n', str(test_f1m_svm) + ', \n')

    f = open('results_svm.txt', 'w')
    f.write(''.join(lines))
    f.close()
# %%
# Read existing results file
if not is_local:
    if os.path.isfile('results_rf.txt'):
        f = open('results_rf.txt', 'r')
        lines = f.readlines()
        f.close()
    else:
        lines = [x + ', \n' for x in ['seed', *annotation_class_names, 'overall_acc', 'macro_f1']]

    # Process files
    lines[0] = lines[0].replace('\n', str(seed) + ', \n')
    for cls in range(n_classes):
        lines[cls + 1] = lines[cls + 1].replace('\n', str(test_f1_rf[cls]) + ', \n')
    lines[n_classes + 1] = lines[n_classes + 1].replace('\n', str(test_acc_rf) + ', \n')
    lines[n_classes + 2] = lines[n_classes + 2].replace('\n', str(test_f1m_rf) + ', \n')

    f = open('results_rf.txt', 'w')
    f.write(''.join(lines))
    f.close()
# %%
## save models
if not is_local:
    import pickle

    with open(f'svm_model_{seed}.pt', 'wb') as f:
        pickle.dump(svm_model, f)
    with open(f'rf_model_{seed}.pt', 'wb') as f:
        pickle.dump(rf_model, f)
# %%
dataset_train.close()
dataset_val.close()
dataset_test.close()