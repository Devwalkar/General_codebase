""" Template Configuration file for CIFAR10 training on Resnet18
"""

# DL model Architecture Settings

'''
Choose DL model from "InceptionV3", "Xception", "VGG_19", "Resnet18", "Resnet34", "Resnet52", "Resnet101", "Resnet152"
                     "DenseNet121", "MobilenetV2","ResNeXt101-32","ResNeXt101-64"

'''
model = dict(
        name="Resnet18",
        pretrained= "imagenet",   # Select between "imagenet" and None
        DataParallel = False,      # Select between breaking single model onto
        Multi_GPU_replica = False  # multiple GPUs or replicating model on 
                                    # multiple GPUs.Only select either of them
        )


# Dataset Settings

'''
Choose dataset from "MNIST", "CIFAR10", "CIFAR100", "Fashion-MNIST"
                    "SVHN", "STL10", "Caltech", "Imagenet"
'''

dataset_cfg = dict(
    id_cfg=dict(
        root= "../data",
        name= "CIFAR10",
        num_classes= 10,
        download= False   # Keep true to download dataset through torch API
    ),
    train_cfg=dict(
        batch_size=128,
        shuffle=True,
        num_workers=8
    ),
    val_cfg=dict(
        batch_size=128,
        shuffle=False,
        num_workers=8
    )
)

# Model Training Settings

train_cfg = dict(
    optimizer=dict(
        name='Adam',
        lr=0.001,
        weight_decay=1e-5
    ),
    criterion=dict(
        L1='CrossEntropyLoss',
    ),
    scheduler=dict(
        name='ReduceLROnPlateau',    # Select from LambdaLR, StepLR, MultiStepLR, 
                                     # ExponentialLR, ReduceLROnPlateau, CylicLR
        patience=2,
        #step_size=15,
        #exp_gamma=0.1,
        verbose=True
    ),

    test_interval = 1,
    plot_accuracy_graphs=True,
    epochs=20,
    gpu=[0,1],
    training_store_root="../Model_storage"
)


# Training Resume settings
# Select from either resuming training or validating model on test set 

Train_resume = False
Validate_only = False
Load_run_id = '01_03_23_49'
Load_Epoch = 3