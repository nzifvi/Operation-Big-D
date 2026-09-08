import torch

class Encoder(torch.nn.Module):
    def __init__(self, latentDim:int, layersParams:list, imgHeight:int, imgWidth:int):
        super(Encoder, self).__init__()
        self._c1 = torch.nn.Conv2d(
            in_channels  = layersParams[0][0],
            out_channels = layersParams[0][1],
            kernel_size  = layersParams[0][2],
            padding      = layersParams[0][3],
            stride       = layersParams[0][4], # downsampling done in conv2d layers, rather than using pooling, to save memory
        )
        self._c2 = torch.nn.Conv2d(
            in_channels  = layersParams[1][0],
            out_channels = layersParams[1][1],
            kernel_size  = layersParams[1][2],
            padding      = layersParams[1][3],
            stride       = layersParams[1][4],
        )
        self._c3 = torch.nn.Conv2d(
            in_channels  = layersParams[2][0],
            out_channels = layersParams[2][1],
            kernel_size  = layersParams[2][2],
            padding      = layersParams[2][3],
            stride       = layersParams[2][4],
        )
        outputDimensions = self._calculateOutputDimensions(imgHeight, imgWidth, layersParams)
        self._fc = torch.nn.Linear(
            in_features = layersParams[2][1] * outputDimensions[-1][0] * outputDimensions[-1][1],
            out_features = latentDim,
        )

        self._preEmbeddingChannels = layersParams[2][1]
        self._preEmbeddingHeight   = outputDimensions[-1][0]
        self._preEmbeddingWidth    = outputDimensions[-1][1]

    def forward(self, x):
        x = torch.nn.functional.relu(self._c1(x))
        x = torch.nn.functional.relu(self._c2(x))
        x = torch.nn.functional.relu(self._c3(x))
        return self._fc(torch.flatten(x, 1))

    def _calculateOutputDimensions(self, imgHeight:int, imgWidth:int, layersParams:list) -> list:
        def calculateOutputDimension(dimensionIn:int, padding:int, kernelSize:int, stride:int) -> int:
            return (dimensionIn + 2*padding - kernelSize) // stride + 1

        dimensions = []
        for i, layerParam in enumerate(layersParams):
            _, _, kernelSize, padding, stride = layerParam
            heightOut = calculateOutputDimension(imgHeight, padding, kernelSize, stride)
            widthOut  = calculateOutputDimension(imgWidth, padding, kernelSize, stride)
            dimensions.append((heightOut, widthOut))
            imgHeight = heightOut
            imgWidth  = widthOut
        return dimensions

    def getPreEmbeddingDimensions(self) -> tuple:
        return self._preEmbeddingChannels, self._preEmbeddingHeight, self._preEmbeddingWidth

class Decoder(torch.nn.Module):
    def __init__(self, latentDim:int, layersParams:list, imgHeight, imgWidth):
        super(Decoder, self).__init__()
        self._fc = torch.nn.Linear(
            in_features = latentDim,
            out_features = layersParams[0][0] * imgHeight * imgWidth
        )
        self._ct1 = torch.nn.ConvTranspose2d(
            in_channels    = layersParams[0][0],
            out_channels   = layersParams[0][1],
            kernel_size    = layersParams[0][2],
            padding        = layersParams[0][3],
            stride         = layersParams[0][4],
            output_padding = layersParams[0][5]
        )
        self._ct2 = torch.nn.ConvTranspose2d(
            in_channels    = layersParams[1][0],
            out_channels   = layersParams[1][1],
            kernel_size    = layersParams[1][2],
            padding        = layersParams[1][3],
            stride         = layersParams[1][4],
            output_padding = layersParams[1][5]
        )
        self._ct3 = torch.nn.ConvTranspose2d(
            in_channels    = layersParams[2][0],
            out_channels   = layersParams[2][1],
            kernel_size    = layersParams[2][2],
            padding        = layersParams[2][3],
            stride         = layersParams[2][4],
            output_padding = layersParams[2][5]
        )

        self._initialChannel = layersParams[0][0]
        self._initialHeight  = imgHeight
        self._initialWidth   = imgWidth

    def forward(self, x):
        x = self._fc(x)
        x = x.view(-1, self._initialChannel, self._initialHeight, self._initialWidth)
        x = torch.nn.functional.relu(self._ct1(x))
        x = torch.nn.functional.relu(self._ct2(x))
        return self._ct3(x)

class Autoencoder(torch.nn.Module):
    def __init__(self, latentDim, encoderParams, decoderParams, imgHeight, imgWidth):
        super(Autoencoder, self).__init__()
        self._encoder = Encoder(latentDim, encoderParams, imgHeight, imgWidth)
        _, preEmbeddingHeight, preEmbeddingWidth = self._encoder.getPreEmbeddingDimensions()
        # add a check that decoderParams[0][0] == preEmbeddingChannels ?
        # - decoder will throw a runtime error if so
        self._decoder = Decoder(latentDim, decoderParams, preEmbeddingHeight, preEmbeddingWidth)

    def forward(self, x):
        z = self._encoder(x)
        return self._decoder(z)

    def encode(self, x):
        return self._encoder(x)

    def decode(self, z):
        return self._decoder(z)