
args['batchSize'] = 64
args['lrStart'] = 0.02
args['lrEnd'] = 0.02
args['nUnits'] = 1024
args['nBatch'] = 10000 #3000
args['nLayers'] = 5
args['seed'] = 0
args['nClasses'] = 40
args['nInputFeatures'] = 256
args['dropout'] = 0.4
args['whiteNoiseSD'] = 0.8
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = True
args['l2_decay'] = 1e-5

modelName = 'repro_old_letters_10layers'

args = {}
args['outputDir'] = '/mnt/scratch/kudrinsk/eval_challenge/trains/' + modelName
args['datasetPath'] = '/mnt/scratch/kudrinsk/eval_challenge/ptDecoder_ctc'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)

def train_model(strideLen, kernelLen, gaussianSmoothWidth):

    return val

from bayes_opt import BayesianOptimization

# Bounded region of parameter space
pbounds = {
    'strideLen': (2, 10),
    'kernelLen': (16, 64),
    'gaussianSmoothWidth': (0.1, 5.0)
    'lrEnd': (0.02, )
}

optimizer = BayesianOptimization(
    f=train_model,
    pbounds=pbounds,
    random_state=1,
)

optimizer.maximize(
    init_points=5,  # Random exploration steps
    n_iter=25,      # Steps of gaussian process
)
