import torch
import numpy
import optuna
import Transforms
import matplotlib.pyplot as plt
from Autoencoder import Autoencoder

"""
Tests to implement:
1) RE as a function of PWN plot

Features to implement:
1) Ability to set seed number for torch generators
"""

class TextureSet(torch.utils.data.Dataset):
    def __init__(self, filePath, transform = None, transformMode = None):
        super(TextureSet, self).__init__()
        data = numpy.load(filePath)
        imgs = numpy.stack([data[name].astype(numpy.uint8) for name in data.files])

        # assumption: the 20 channels per image does not matter and can be collapsed to 1
        # - approach disregards all other 19.
        # - information may potentially be lost if the 20 channels are not colour channels.
        self._transform = transform
        self._transformMode = transformMode
        if self._transform is not None and self._transformMode == "onInitialisation":
            self._imgTensor = self._transform(torch.from_numpy(imgs[0, :, 0, :, :]))
        else:
            self._imgTensor = torch.from_numpy(imgs[0, :, 0, :, :])

    def __len__(self):
        return self._imgTensor.shape[0]

    def __getitem__(self, index):
        if self._transform is not None:
            if self._transformMode == "onAccess":
                return self._transform(self._imgTensor[index].float().unsqueeze(0))
            elif self._transformMode == "onInitialisation":
                return self._imgTensor[index].float().unsqueeze(0)
            else:
                raise ValueError(f"{self._transformMode} is not a supported transformation mode")
        else:
            return self._imgTensor[index].float().unsqueeze(0)

def train(model, trainLoader, epochCount, lossFunction, optimiser, device) -> tuple:
    model.train()
    epochREs = []
    for e in range(epochCount):
        epochRE = 0.0
        for x in trainLoader:
            x = x.to(device, non_blocking = True)
            optimiser.zero_grad()
            yReconstructed = model(x)
            re = lossFunction(yReconstructed, x)
            re.backward()
            optimiser.step()
            epochRE += re.item()
        epochREs.append(epochRE / len(trainLoader))
    return model, epochREs

def validate(model, validationLoader, lossFunction, device) -> float:
    model.eval()
    with torch.no_grad():
        reSum = 0.0
        for x in validationLoader:
            x = x.to(device, non_blocking = True)
            yReconstructed = model(x)
            re = lossFunction(x, yReconstructed)
            reSum += re.item()

    return reSum / len(validationLoader)

def optimise() -> None:
    def objective(trial):
        configIndex = trial.suggest.categorical("configIndex", list(range(len(encoderPossibleConfigs))))
        latentDim = trial.suggest.categorical("latentDim", possibleLatentDims)
        lossFuncIndex = trial.suggest.categorical("lossFuncIndex", list(range(len(possibleLossFunctions))))
        optimiserIndex = trial.suggest.categorical("optimiserIndex", list(range(len(possibleOptimisers))))
        learningRate = trial.suggest.categorical("learningRate", possibleLearningRates)

        autoencoder = Autoencoder(
            latentDim=latentDim,
            encoderParams=encoderPossibleConfigs[configIndex],
            decoderParams=decoderPossibleConfigs[configIndex],
            imgHeight=textureSet[0].shape[0],
            imgWidth=textureSet[0].shape[1]
        )
        optimiser = possibleOptimisers[optimiserIndex](autoencoder.parameters(), learningRate)
        lossFunction = possibleLossFunctions[lossFuncIndex]

        retrialAmount = 5
        errorSum = 0.0
        for i in range(retrialAmount):
            autoencoder, _ = train(
                model=autoencoder,
                trainLoader=trainLoader,
                epochCount=30,
                lossFunction=lossFunction,
                optimiser=optimiser,
            )
            errorSum += validate(
                model=autoencoder,
                validationLoader=validationLoader,
                lossFunction=lossFunction,
            )
        return errorSum / retrialAmount
    textureSet = TextureSet(
        filePath = "data/Images_data_15.npz"
    )
    trainSet, validationSet = torch.utils.data.random_split(
        textureSet,
        [int(len(textureSet) * 0.7), int(len(textureSet) * 0.3)]
    )
    trainLoader = torch.utils.data.DataLoader(
        trainSet,
        batch_size = 32,
        shuffle = True,
    )
    validationLoader = torch.utils.data.DataLoader(
        validationSet,
        batch_size = 32,
        shuffle = False
    )
    # tuple structure = (input_channels, output_channels, kernel_size, padding, stride)
    # - each possible configuration must have 3 tuples in the above format
    # - ith possible encoder configuration must correspond with ith possible decoder configuration in such a way to form a bottleneck architecture
    encoderPossibleConfigs = [
        [
            (1, 8, 3, 0, 2),
            (8, 16, 3, 0, 2),
            (16, 32, 3, 0, 2)
        ]
    ]
    # tuple structure = (input_channels, output_channels, kernel_size, padding, stride, output_padding)
    # - each possible configuration must have 3 tuples in the above format
    # - ith possible decoder configuration must correspond with ith possible encoder configuration in such a way to form a bottleneck architecture
    decoderPossibleConfigs = [
        [
            (32, 16, 3, 0, 2, (1, 0)),
            (16, 8, 3, 0, 2, (1, 0)),
            (8, 1, 3, 0, 2, (1, 1))
        ]
    ]
    possibleLatentDims    = [8, 16, 32, 64, 128, 256]
    possibleLossFunctions = [torch.nn.MSELoss()]
    possibleOptimisers    = [torch.optim.Adam, torch.optim.RMSprop, torch.optim.AdamW]
    possibleLearningRates = [0.001, 0.005, 0.01, 0.05]

    study = optuna.create_study(direction = "minimize")
    study.optimize(
        objective,
        n_trials = len(encoderPossibleConfigs) * len(possibleLatentDims) * len(possibleLossFunctions) * len(possibleOptimisers) * len(possibleLearningRates),
    )

def testREOverlatentDim(loaders, device, retrialAmount, dims, modelParams, imgShape, lossFunc, optimiser, epochCount) -> list:
    trainLoader, valLoader       = loaders
    report = []
    for latentDim in dims:
        RESum = 0.0
        for i in range(retrialAmount):
            autoencoder = Autoencoder(
                latentDim     = latentDim,
                encoderParams = modelParams[0],
                decoderParams = modelParams[1],
                imgHeight     = imgShape[0],
                imgWidth      = imgShape[1]
            )
            autoencoder, _  = train(
                model        = autoencoder,
                trainLoader  = trainLoader,
                epochCount   = epochCount,
                lossFunction = lossFunc,
                optimiser    = optimiser,
                device       = device
            )
            RESum += validate(
                model            = autoencoder,
                validationLoader = validationLoader,
                lossFunction     = lossFunc,
                device           = device
            )
        report.append(
            (RESum / retrialAmount, latentDim)
        )
    return report

def testREOverPWN(loaders, device, retrialAmount, dim, modelParams, imgShape, lossFunc, optimiser, epochCount) -> list:
    ...

def testREOverNoise(loaders, device, retrialAmount, dim, modelParams, imgShape, lossFunc, optimiser, epochCount) -> list:
    ...



device = torch.device("cuda")
autoencoder = Autoencoder(
    latentDim = 8,
    encoderParams = [
        (1, 8, 3, 0, 2),
        (8, 16, 3, 0, 2),
        (16, 32, 3, 0, 2)
    ],
    decoderParams = [
        (32, 16, 3, 0, 2, (1, 0)),
        (16, 8, 3, 0, 2, (1, 0)),
        (8, 1, 3, 0, 2, (1, 1))
    ],
    imgHeight = 110,
    imgWidth = 120,
)
autoencoder = autoencoder.to(device)
textureSet = TextureSet(
    filePath = "data/Images_data_15.npz",
    transform = Transforms.apply1StageTransformation,
    transformMode = "onInitialisation"
)
trainSet, validationSet = torch.utils.data.random_split(
    textureSet,
    [int(len(textureSet) * 0.7), int(len(textureSet) * 0.3)]
)
trainLoader = torch.utils.data.DataLoader(
    trainSet,
    batch_size = 32,
    shuffle = True,
)
validationLoader = torch.utils.data.DataLoader(
    validationSet,
    batch_size = 32,
    shuffle = False
)



