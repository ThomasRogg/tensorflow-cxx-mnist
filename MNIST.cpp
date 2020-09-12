// -----------------------------------------------------------------------------
//  MNIST.c - see http://yann.lecun.com/exdb/mnist/
// -----------------------------------------------------------------------------

#define _CRT_SECURE_NO_WARNINGS
#define WIN32_LEAN_AND_MEAN

#include <windows.h>
#include <winsock.h>

// undefine min/max from windows.h, required for TensorFlow
#undef min
#undef max

#pragma warning(push, 0)
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/client/client_session.h"

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/state_ops.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#pragma warning(pop)

#include <vector>
#include <utility>

#include <cstdlib>
#include <cstdio>
#include <ctime>

using namespace std;
using namespace tensorflow;


// Global variables
vector<pair<Input::Initializer, Input::Initializer>>            gTrainBatches;
unique_ptr<pair<Input::Initializer, Input::Initializer>>        gValidateSet, gTestSet;

vector<Output>                                                  gInitializers, gWeightOutputs;
vector<TensorShape>                                             gWeightOutputsShape;

vector<Output>                                                  gSaveWeights, gRestoreWeights;

// Constant variables
#define MAX_NUM_TRAIN                   0       // set to 0 for unlimited
#define MAX_NUM_TEST                    0

#define IMAGE_WIDTH                     28
#define IMAGE_HEIGHT                    28
#define IMAGE_PIXELS                    (IMAGE_WIDTH * IMAGE_HEIGHT)

#define NUM_HIDDEN_LAYERS               4
#define NUM_NEURONS_PER_HIDDEN_LAYER    IMAGE_PIXELS

#define BATCH_SIZE                      128

#define VALIDATE_COUNT_OF_TRAIN         0.2
#define MAX_EPOCH_NO_SUCCESS            120


// -----------------------------------------------------------------------------
//  LoadSet - load set file
// -----------------------------------------------------------------------------

bool LoadSet(const char *fNamePart, unsigned int maxCount, bool isTrain)
{
    float* input = NULL, * output;
    float* in_ptr, * out_ptr;
    int* indices = NULL;
    char txt[80];

    sprintf(txt, "data/%s-labels-idx1-ubyte", fNamePart);
    FILE* fLabels = fopen(txt, "rb");
    if(!fLabels)
    {
        fprintf(stderr, "Cannot open '%s' for reading!\n", txt);
        return false;
    }

    sprintf(txt, "data/%s-images-idx3-ubyte", fNamePart);
    FILE* fImages = fopen(txt, "rb");
    if(!fImages)
    {
        fprintf(stderr, "Cannot open '%s' for reading!\n", txt);

        fclose(fLabels);
        return false;
    }

    unsigned int headerLabels[2], headerImages[4], count;
    if(fread(headerLabels, 4, 2, fLabels) != 2
    || fread(headerImages, 4, 4, fImages) != 4)
    {
        fprintf(stderr, "End of file error!\n");
        goto err;
    }

    count = ntohl(headerLabels[1]);
    if(ntohl(headerLabels[0]) != 0x00000801
    || ntohl(headerImages[0]) != 0x00000803
    || !count
    || count != ntohl(headerImages[1])
    || ntohl(headerImages[2]) != IMAGE_WIDTH
    || ntohl(headerImages[3]) != IMAGE_HEIGHT)
    {
        fprintf(stderr, "Header file format error!\n");
        goto err;
    }
    if(maxCount && count > maxCount)
        count = maxCount;

    input = in_ptr = (float*)malloc((IMAGE_PIXELS + 10) * count * sizeof(float));
    if (!input)
    {
        fprintf(stderr, "Memory full!\n");
        goto err;
    }
    output = out_ptr = input + IMAGE_PIXELS * count;

    if (isTrain)
    {
        indices = (int*)malloc(count * sizeof(int));
        if (!indices)
        {
            free(input);

            fprintf(stderr, "Memory full!\n");
            goto err;
        }

        for (unsigned int i = 0; i < count; i++)
            indices[i] = i;
        random_shuffle(indices, indices + count);
    }

    for (unsigned int i = 0; i < count; i++)
    {
        unsigned char data[IMAGE_PIXELS];
        unsigned char label;

        if (fread(data, IMAGE_PIXELS, 1, fImages) != 1
        || fread(&label, 1, 1, fLabels) != 1)
        {
            fprintf(stderr, "End of file error!\n");
            goto err;
        }
        if (label >= 10)
        {
            fprintf(stderr, "Label not 0-9!\n");
            goto err;
        }

        if (isTrain)
        {
            in_ptr = input + indices[i] * IMAGE_PIXELS;
            out_ptr = output + indices[i] * 10;
        }
        for (int j = 0; j < IMAGE_PIXELS; j++)
            *in_ptr++ = data[j] / 255.0f - 0.5f;
        for (int j = 0; j < 10; j++)
            *out_ptr++ = j == label ? 1.0f : 0.0f;
    }

    if (isTrain)
    {
        unsigned int validateCount = (unsigned int)(count * VALIDATE_COUNT_OF_TRAIN);
        count -= validateCount;

        in_ptr = input + count * IMAGE_PIXELS;
        out_ptr = output + count * 10;
        gValidateSet = make_unique<pair<Input::Initializer, Input::Initializer>>(
            Input::Initializer(initializer_list<float>(in_ptr, in_ptr + validateCount * IMAGE_PIXELS), { validateCount, IMAGE_PIXELS }),
            Input::Initializer(initializer_list<float>(out_ptr, out_ptr + validateCount * 10), { validateCount, 10 }));

        for (unsigned int i = 0; i < count; i += BATCH_SIZE)
        {
            unsigned int size = BATCH_SIZE;
            if (size > count - i)
                size = count - i;

            in_ptr = input + i * IMAGE_PIXELS;
            out_ptr = output + i * 10;
            gTrainBatches.push_back(make_pair(
                Input::Initializer(initializer_list<float>(in_ptr, in_ptr + size * IMAGE_PIXELS), { size, IMAGE_PIXELS }),
                Input::Initializer(initializer_list<float>(out_ptr, out_ptr + size * 10), { size, 10 })));
        }
    }
    else
        gTestSet = make_unique<pair<Input::Initializer, Input::Initializer>>(
            Input::Initializer(initializer_list<float>(input, input + count * IMAGE_PIXELS), { count, IMAGE_PIXELS }),
            Input::Initializer(initializer_list<float>(output, output + count * 10), { count, 10 }));

    free(input);
    free(indices);
    fclose(fLabels);
    fclose(fImages);
    return true;

err:
    free(input);
    free(indices);
    fclose(fLabels);
    fclose(fImages);
    return false;
}


// -----------------------------------------------------------------------------
//  AdamOptimizer
//  Implemention of https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/optimizer_v2/adam.py
// -----------------------------------------------------------------------------

vector<Output> AdamOptimizer(Scope &scope, Output loss, float learningRate, float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1E-7)
{
    vector<Output> grads;
    vector<Output> optimizers;

    Status s = AddSymbolicGradients(scope, {loss}, gWeightOutputs, &grads);
    if(!s.ok())
        fprintf(stderr, "Error: %s\n", s.ToString().c_str());

    for (int i = 0; i < gWeightOutputs.size(); i++)
    {
        auto m = ops::Variable(scope, gWeightOutputsShape[i], DT_FLOAT);
        auto mInitial = ops::ZerosLike(scope, m);
        gInitializers.push_back(ops::Assign(scope, m, mInitial));

        auto v = ops::Variable(scope, gWeightOutputsShape[i], DT_FLOAT);
        auto vInitial = ops::ZerosLike(scope, v);
        gInitializers.push_back(ops::Assign(scope, v, vInitial));

        optimizers.push_back(ops::ApplyAdam(scope, gWeightOutputs[i], m, v, 0.0f, 0.0f, learningRate, beta1, beta2, epsilon, grads[i]));
    }

    return optimizers;
}


// -----------------------------------------------------------------------------
//  AddFullyConnectedLayer
// -----------------------------------------------------------------------------

Output AddFullyConnectedLayer(Scope& scope, Output layer, int lastNumNeurons, int numNeurons)
{
    initializer_list<ptrdiff_t> weightsShape = { lastNumNeurons, numNeurons };
    auto weightsInitial = ops::RandomNormal(scope, ops::Const(scope, weightsShape), DT_FLOAT);
    auto weights = ops::Variable(scope, weightsShape, DT_FLOAT);

    gInitializers.push_back(ops::Assign(scope, weights, weightsInitial));
    gWeightOutputs.push_back(weights);
    gWeightOutputsShape.push_back(TensorShape(weightsShape));

    auto weightsSave = ops::Variable(scope, weightsShape, DT_FLOAT);
    gSaveWeights.push_back(ops::Assign(scope, weightsSave, weights));
    gRestoreWeights.push_back(ops::Assign(scope, weights, weightsSave));

    initializer_list<ptrdiff_t> biasShape = { numNeurons };
    auto biasInitial = ops::RandomNormal(scope, ops::Const(scope, biasShape), DT_FLOAT);
    auto bias = ops::Variable(scope, biasShape, DT_FLOAT);

    auto biasSave = ops::Variable(scope, biasShape, DT_FLOAT);
    gSaveWeights.push_back(ops::Assign(scope, biasSave, bias));
    gRestoreWeights.push_back(ops::Assign(scope, bias, biasSave));

    gInitializers.push_back(ops::Assign(scope, bias, biasInitial));
    gWeightOutputs.push_back(bias);
    gWeightOutputsShape.push_back(TensorShape(biasShape));

    layer = ops::Add(scope, ops::MatMul(scope, layer, weights), bias);
    return ops::Relu(scope, layer);
}


// -----------------------------------------------------------------------------
//  BuildGraph
// -----------------------------------------------------------------------------

pair<Output, vector<Output>> BuildGraph(Scope& scope, ops::Placeholder &image, ops::Placeholder &label)
{
    Output layer = image;
    for (int i = 0; i < NUM_HIDDEN_LAYERS; i++)
        layer = AddFullyConnectedLayer(scope, layer, i == 0 ? IMAGE_PIXELS : NUM_NEURONS_PER_HIDDEN_LAYER, NUM_NEURONS_PER_HIDDEN_LAYER);

    {
        initializer_list<ptrdiff_t> weightsShape = { NUM_HIDDEN_LAYERS == 0 ? IMAGE_PIXELS : NUM_NEURONS_PER_HIDDEN_LAYER, 10 };
        auto weightsInitial = ops::RandomNormal(scope, ops::Const(scope, weightsShape), DT_FLOAT);
        auto weights = ops::Variable(scope, weightsShape, DT_FLOAT);

        gInitializers.push_back(ops::Assign(scope, weights, weightsInitial));
        gWeightOutputs.push_back(weights);
        gWeightOutputsShape.push_back(TensorShape(weightsShape));

        auto weightsSave = ops::Variable(scope, weightsShape, DT_FLOAT);
        gSaveWeights.push_back(ops::Assign(scope, weightsSave, weights));
        gRestoreWeights.push_back(ops::Assign(scope, weights, weightsSave));

        initializer_list<ptrdiff_t> biasShape = { 10 };
        auto biasInitial = ops::RandomNormal(scope, ops::Const(scope, biasShape), DT_FLOAT);
        auto bias = ops::Variable(scope, biasShape, DT_FLOAT);

        auto biasSave = ops::Variable(scope, biasShape, DT_FLOAT);
        gSaveWeights.push_back(ops::Assign(scope, biasSave, bias));
        gRestoreWeights.push_back(ops::Assign(scope, bias, biasSave));

        gInitializers.push_back(ops::Assign(scope, bias, biasInitial));
        gWeightOutputs.push_back(bias);
        gWeightOutputsShape.push_back(TensorShape(biasShape));

        layer = ops::Add(scope, ops::MatMul(scope, layer, weights), bias);
    }

    auto prediction = ops::Equal(scope, ops::ArgMax(scope, layer, 1), ops::ArgMax(scope, label, 1));
    auto success = ops::Mean(scope, ops::Cast(scope, prediction, DT_FLOAT), -1);

    auto loss = ops::Mean(scope, ops::SoftmaxCrossEntropyWithLogits(scope, layer, label).loss, -1);
    auto optimizers = AdamOptimizer(scope, loss, 0.001f);

    return make_pair(success, optimizers);
}


// -----------------------------------------------------------------------------
//  main - program entry point
// -----------------------------------------------------------------------------

int main()
{
    printf("Loading...\n");
    srand((unsigned int)time(NULL));

    if(!LoadSet("train", MAX_NUM_TRAIN, true)
    || !LoadSet("t10k", MAX_NUM_TEST, false))
        return EXIT_FAILURE;

    Scope scope = Scope::NewRootScope();
    auto image = ops::Placeholder(scope, DT_FLOAT, ops::Placeholder::Shape({ -1, IMAGE_PIXELS }));
    auto label = ops::Placeholder(scope, DT_FLOAT, ops::Placeholder::Shape({ -1, 10 }));
    auto res = BuildGraph(scope, image, label);
    auto &success = res.first;
    auto &optimizers = res.second;

    ClientSession session(scope);
    Status s;

    s = session.Run(gInitializers, NULL);
    if (!s.ok())
    {
        fprintf(stderr, "Error: %s\n", s.ToString().c_str());
        return EXIT_FAILURE;
    }

    vector<Tensor> outputs;
    float bestSuccess = -1;
    int bestSuccessEpoch = 0;

    for(int e = 0;; e++)
    {
        s = session.Run({ {image, gValidateSet->first}, {label, gValidateSet->second} }, { success }, &outputs);
        if (!s.ok())
        {
            fprintf(stderr, "Error: %s\n", s.ToString().c_str());
            return EXIT_FAILURE;
        }

        float success = outputs[0].scalar<float>()();
        printf("Epoch #%d   Validation Success: %.2f %%\n", e, success * 100);

        if (bestSuccess < success)
        {
            s = session.Run(gSaveWeights, NULL);
            if (!s.ok())
            {
                fprintf(stderr, "Error: %s\n", s.ToString().c_str());
                return EXIT_FAILURE;
            }

            bestSuccess = success;
            bestSuccessEpoch = e;
        }
        else if(e - bestSuccessEpoch >= MAX_EPOCH_NO_SUCCESS)
            break;

        for (int b = 0; b < gTrainBatches.size(); b++)
        {
            s = session.Run({ {image, gTrainBatches[b].first}, {label, gTrainBatches[b].second} }, optimizers, NULL);
            if (!s.ok())
            {
                fprintf(stderr, "Error: %s\n", s.ToString().c_str());
                return EXIT_FAILURE;
            }
        }
    }

    s = session.Run(gRestoreWeights, NULL);
    if (!s.ok())
    {
        fprintf(stderr, "Error: %s\n", s.ToString().c_str());
        return EXIT_FAILURE;
    }

    s = session.Run({ {image, gTestSet->first}, {label, gTestSet->second} }, { success }, &outputs);
    if (!s.ok())
    {
        fprintf(stderr, "Error: %s\n", s.ToString().c_str());
        return EXIT_FAILURE;
    }
    {
        float success = outputs[0].scalar<float>()();
        printf("Done, after %d epochs!   Test Success: %.2f %%\n", bestSuccessEpoch, success * 100);
    }

    return EXIT_SUCCESS;
}